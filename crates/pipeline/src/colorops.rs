//! Colour tools in OkLCh on display-linear Rec.2020 values: vibrance, saturation, the 8-band colour
//! mixer, B&W mix, and 3-way colour grading; plus the camera-calibration matrix (scene linear).

use std::f32::consts::{PI, TAU};
use std::sync::{Arc, OnceLock};

use lcc_color::perceptual::{hsv_to_rgb, lab_to_lch, lch_to_lab, oklab_from_2020, oklab_to_2020};
use lcc_color::{D50, D65, Mat3, PROPHOTO, REC2020, SRGB, Xy, bradford};
use lcc_develop::{Calibration, DevelopSettings, MIXER_HUES, PointColor};

/// OkLCh hue angle (radians) of a pure sRGB colour with HSV hue `deg`.
pub fn oklch_hue_of_srgb_hue(deg: f64) -> f32 {
    let c = hsv_to_rgb(deg as f32, 1.0, 1.0);
    let lin = c.map(lcc_color::transfer::srgb_to_linear);
    let m = SRGB.to_space(&REC2020).apply_f32(lin);
    lab_to_lch(oklab_from_2020(m))[2]
}

/// OkLCh hue (radians) of each colour-mixer band centre.
pub fn band_hues() -> &'static [f32; 8] {
    static H: OnceLock<[f32; 8]> = OnceLock::new();
    H.get_or_init(|| MIXER_HUES.map(oklch_hue_of_srgb_hue))
}

#[inline]
fn wrap(a: f32) -> f32 {
    (a + PI).rem_euclid(TAU) - PI
}

/// Partition-of-unity weights of hue `h` over the 8 bands (raised cosine between neighbours).
#[inline]
pub fn band_weights(h: f32) -> [f32; 8] {
    let hues = band_hues();
    let mut w = [0.0f32; 8];
    for i in 0..8 {
        let a = hues[i];
        let b = hues[(i + 1) % 8];
        let span = wrap(b - a).rem_euclid(TAU);
        let d = wrap(h - a).rem_euclid(TAU);
        if d <= span {
            let t = d / span;
            let s = 0.5 - 0.5 * (t * PI).cos();
            w[i] += 1.0 - s;
            w[(i + 1) % 8] += s;
            break;
        }
    }
    w
}

/// A colour-grading wheel as an OkLab offset (a, b) and a lightness shift.
#[derive(Clone, Copy, Debug)]
pub struct WheelK {
    pub a: f32,
    pub b: f32,
    pub lum: f32,
}

/// Number of `f32`s per Point Color sample in [`PointK::words`] (the GPU parameter layout).
pub const POINT_WORDS: usize = 10;

/// A Point Color sample resolved for the per-pixel stage: the sample (OkLab lightness, chroma, hue
/// in radians), the adjustment (hue rotation in radians, chroma scale − 1, lightness shift,
/// variance factor − 1) and the half-widths of the range box (hue radians, chroma, lightness).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointK {
    pub l: f32,
    pub c: f32,
    pub h: f32,
    pub dh: f32,
    pub sat: f32,
    pub dl: f32,
    pub var: f32,
    pub wh: f32,
    pub wc: f32,
    pub wl: f32,
}

impl PointK {
    pub fn new(p: &PointColor) -> PointK {
        let k = (p.range.clamp(0.0, 100.0) / 50.0).max(0.05) as f32;
        let f = |v: f64| (v.clamp(0.0, 100.0) / 100.0) as f32;
        PointK {
            l: p.lum as f32,
            c: p.chroma.max(0.0) as f32,
            h: (p.hue as f32).to_radians(),
            dh: (p.hue_shift.clamp(-100.0, 100.0) / 100.0) as f32 * 0.5,
            sat: (p.sat_shift.clamp(-100.0, 100.0) / 100.0) as f32,
            dl: (p.lum_shift.clamp(-100.0, 100.0) / 100.0) as f32 * 0.2,
            var: (p.variance.clamp(-100.0, 100.0) / 100.0) as f32 * 0.8,
            wh: (0.1 + 0.6 * f(p.hue_range)) * k,
            wc: (0.02 + 0.12 * f(p.sat_range)) * k,
            wl: (0.05 + 0.4 * f(p.lum_range)) * k,
        }
    }

    pub fn words(&self) -> [f32; POINT_WORDS] {
        [self.l, self.c, self.h, self.dh, self.sat, self.dl, self.var, self.wh, self.wc, self.wl]
    }

    /// How much an OkLCh colour belongs to this sample's range (0..1, soft edges).
    #[inline]
    pub fn weight(&self, l: f32, c: f32, h: f32) -> f32 {
        // hue is meaningless for near-neutral colours: a chromatic sample ignores them, a
        // near-neutral sample selects by chroma and lightness only
        let wh = if self.c < 0.02 { 1.0 } else { (1.0 - smooth(0.5 * self.wh, self.wh, wrap(h - self.h).abs())) * smooth(0.005, 0.025, c) };
        if wh <= 0.0 {
            return 0.0;
        }
        let wc = 1.0 - smooth(0.5 * self.wc, self.wc, (c - self.c).abs());
        let wl = 1.0 - smooth(0.5 * self.wl, self.wl, (l - self.l).abs());
        wh * wc * wl
    }

    /// Apply this sample's adjustment to an OkLCh colour.
    #[inline]
    pub fn apply(&self, lch: [f32; 3]) -> [f32; 3] {
        self.apply_scaled(lch, 1.0)
    }

    /// [`Self::apply`] at `strength` 0..1 (a mask's opacity there: its local Point Color).
    #[inline]
    pub fn apply_scaled(&self, lch: [f32; 3], strength: f32) -> [f32; 3] {
        let [mut l, mut c, mut h] = lch;
        let w = self.weight(l, c, h) * strength;
        if w <= 0.0 {
            return lch;
        }
        h += w * (self.var * wrap(h - self.h) + self.dh);
        c += w * self.var * (c - self.c);
        c = (c * (1.0 + w * self.sat)).max(0.0);
        l += w * (self.var * (l - self.l) + self.dl);
        [l, c, h]
    }
}

/// The Point Color samples of `points` that change something, resolved (at most
/// [`lcc_develop::MAX_POINT_COLORS`]).
pub fn point_ks(points: &[PointColor]) -> Vec<PointK> {
    points.iter().take(lcc_develop::MAX_POINT_COLORS).filter(|p| !p.is_neutral()).map(PointK::new).collect()
}

/// The masks' local Point Color samples, one list per mask the per-pixel stage evaluates (visible,
/// with components: the order of [`crate::masks::evaluate`]); empty when no mask has any, and in a
/// black-and-white rendition (Point Color does nothing there, as globally).
pub fn mask_point_ks(s: &DevelopSettings) -> Vec<Vec<PointK>> {
    if crate::is_bw(s) {
        return Vec::new();
    }
    let v: Vec<Vec<PointK>> = s.masks.iter().filter(|m| m.visible && !m.components.is_empty()).map(|m| point_ks(&m.point_colors)).collect();
    if v.iter().all(Vec::is_empty) { Vec::new() } else { v }
}

/// Words of a resolved colour mixer ([`MixerK::words`]): hue, saturation, luminance of the 8 bands.
pub const MIXER_WORDS: usize = 24;

/// Entries in each HSL-curve LUT ([`HslK`]): hue positions around the OkLCh circle, plenty for a
/// smooth spline read on the GPU as on the CPU.
pub const HSL_LUT_N: usize = 128;
/// f32s the GPU kernel reads for [`HslK`]: the hue, saturation and luminance LUTs in turn.
pub const HSL_WORDS: usize = 3 * HSL_LUT_N;

/// One HSL curve's value at hue position `x` (0..1 around the OkLCh hue circle): the monotone
/// spline through the user's points, with the circle's ends pinned at neutral (0 = red, where the
/// mixer's hue axis also starts). The curve a point anywhere shapes, not fixed colour bands.
pub fn hsl_curve_at(points: &[lcc_geom::Point], x: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let at = |p: &lcc_geom::Point| (p.x, p.y);
    let mut all: Vec<(f64, f64)> = points.iter().map(at).collect();
    all.push((0.0, 0.0));
    all.push((1.0, 0.0));
    lcc_color::spline::MonotoneCurve::new(&all).eval(x.clamp(0.0, 1.0))
}

/// The HSL curves resolved once per render: three LUTs of engine-unit deltas over the hue circle —
/// hue in radians (±0.5 at ±1, the mixer's own hue scale), saturation as a scale (±1), luminance
/// as an OkLab lightness shift (±0.18, the mixer's own). `None` when every curve is empty (and in
/// a black-and-white rendition, as the colour mixer).
#[derive(Clone, Debug)]
pub struct HslK {
    pub hue: [f32; HSL_LUT_N],
    pub sat: [f32; HSL_LUT_N],
    pub lum: [f32; HSL_LUT_N],
}

impl HslK {
    pub fn new(c: &lcc_develop::HslCurves) -> Option<HslK> {
        if c.is_neutral() {
            return None;
        }
        let lut = |points: &[lcc_geom::Point], scale: f64| {
            std::array::from_fn(|i| (hsl_curve_at(points, i as f64 / (HSL_LUT_N - 1) as f64) * scale) as f32)
        };
        Some(HslK { hue: lut(&c.hue, 0.5), sat: lut(&c.sat, 1.0), lum: lut(&c.lum, 0.18) })
    }

    /// The words the GPU kernel reads: hue, saturation and luminance LUTs in turn.
    pub fn words(&self) -> Vec<f32> {
        self.hue.iter().chain(&self.sat).chain(&self.lum).copied().collect()
    }

    /// The three curves' deltas at hue position `x` (0..1), as the per-pixel stage applies them.
    #[inline]
    pub fn at(&self, x: f32) -> [f32; 3] {
        let f = (x.clamp(0.0, 1.0) * (HSL_LUT_N - 1) as f32).min((HSL_LUT_N - 2) as f32);
        let i = f.floor() as usize;
        let t = f - i as f32;
        let lerp = |a: &[f32; HSL_LUT_N]| a[i] + (a[i + 1] - a[i]) * t;
        [lerp(&self.hue), lerp(&self.sat), lerp(&self.lum)]
    }
}

/// A colour mixer resolved for the per-pixel stage: per band, the hue shift (radians), the
/// saturation scale and the luminance shift. The global mixer ([`ColorOps`]) and each mask's local
/// one use the same maths ([`mixer_apply`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MixerK {
    pub hue: [f32; 8],
    pub sat: [f32; 8],
    pub lum: [f32; 8],
}

impl MixerK {
    pub fn new(m: &lcc_develop::Mixer) -> MixerK {
        let bands = m.bands();
        let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
        MixerK {
            hue: bands.map(|b| (fin(b.hue) / 100.0) as f32 * 0.5),
            sat: bands.map(|b| (fin(b.sat) / 100.0) as f32),
            lum: bands.map(|b| (fin(b.lum) / 100.0) as f32 * 0.18),
        }
    }

    /// The words the GPU kernel reads: hue[8], sat[8], lum[8].
    pub fn words(&self) -> [f32; MIXER_WORDS] {
        let mut w = [0.0; MIXER_WORDS];
        for (slot, v) in w.iter_mut().zip(self.hue.iter().chain(&self.sat).chain(&self.lum)) {
            *slot = *v;
        }
        w
    }

    /// OkLCh `lch` through the mixer at `strength` (0..1: a mask's opacity; 1 = the full mixer).
    #[inline]
    pub fn apply_scaled(&self, lch: [f32; 3], strength: f32) -> [f32; 3] {
        mixer_apply(&self.hue, &self.sat, &self.lum, lch, strength)
    }
}

/// The colour mixer's maths on OkLCh `lch`: each band's shifts weighted by the hue's band weights,
/// hue and luminance shifts faded out towards neutrals; `strength` scales the shifts (1 = as set).
#[inline]
pub fn mixer_apply(hue: &[f32; 8], sat: &[f32; 8], lum: &[f32; 8], lch: [f32; 3], strength: f32) -> [f32; 3] {
    let [mut l, mut c, mut h] = lch;
    let w = band_weights(h);
    let (mut dh, mut ds, mut dl) = (0.0, 0.0, 0.0);
    for i in 0..8 {
        dh += w[i] * hue[i];
        ds += w[i] * sat[i];
        dl += w[i] * lum[i];
    }
    let chroma_w = (c / 0.12).min(1.0);
    h += dh * strength * chroma_w;
    c *= (1.0 + ds * strength).max(0.0);
    l += dl * strength * chroma_w * l.max(0.05).sqrt();
    [l, c, h]
}

/// A Color Mixer as a rendering process applies it: V1–V8's band shifts in OkLCh ([`MixerK`]),
/// from V9 Lightroom Classic's measured response ([`crate::mixer_measured`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Mixer {
    Bands(MixerK),
    Measured(Arc<crate::mixer_measured::Measured>),
}

impl Mixer {
    /// The mixer `m` as `process` renders it; `None` when it changes nothing.
    pub fn new(m: &lcc_develop::Mixer, process: lcc_develop::Process) -> Option<Mixer> {
        if m.is_neutral() {
            return None;
        }
        if process >= lcc_develop::Process::V9 {
            let bands = m.bands();
            let fin = |v: f64| if v.is_finite() { (v / 100.0) as f32 } else { 0.0 };
            return crate::mixer_measured::measured(&bands.map(|b| fin(b.hue)), &bands.map(|b| fin(b.sat)), &bands.map(|b| fin(b.lum)))
                .map(Mixer::Measured);
        }
        Some(Mixer::Bands(MixerK::new(m)))
    }
}

/// The masks' local Color Mixers, one per mask the per-pixel stage evaluates (`None`: neutral);
/// empty when no mask has one, and in a black-and-white rendition (as the global mixer).
pub fn mask_mixer_ks(s: &DevelopSettings) -> Vec<Option<Mixer>> {
    if crate::is_bw(s) {
        return Vec::new();
    }
    let process = s.process.process();
    let v: Vec<Option<Mixer>> = s.masks.iter().filter(|m| m.visible && !m.components.is_empty()).map(|m| Mixer::new(&m.mixer, process)).collect();
    if v.iter().all(Option::is_none) { Vec::new() } else { v }
}

/// The colour tools' parameters, resolved once per render (fields are read by the GPU kernel).
#[derive(Clone, Debug)]
pub struct ColorOps {
    pub vibrance: f32,
    pub saturation: f32,
    pub hue: [f32; 8],
    pub sat: [f32; 8],
    pub lum: [f32; 8],
    /// A V1–V8 mixer ([`MixerK`]: `hue`, `sat`, `lum`) changes something.
    pub mixer: bool,
    /// A V9 mixer (applied first, to the linear colour).
    pub measured: Option<Arc<crate::mixer_measured::Measured>>,
    /// Point Color samples that change something (applied after the mixer).
    pub points: Vec<PointK>,
    pub bw: Option<[f32; 8]>,
    /// Wheels (shadows, midtones, highlights, global), blending, balance.
    pub grading: Option<([WheelK; 4], f32, f32)>,
    /// The colourist primaries ([`PrimariesK`]), applied before the colour tools: balance first,
    /// look after.
    pub primaries: Option<PrimariesK>,
    /// The HSL curves ([`HslK`]), applied after the colour mixer on the same OkLCh values.
    pub hsl: Option<HslK>,
    /// OkLCh hue of skin tones (protected by vibrance).
    pub skin: f32,
}

pub fn wheel(w: &lcc_develop::Wheel) -> WheelK {
    let h = oklch_hue_of_srgb_hue(w.hue);
    let s = (w.sat / 100.0) as f32 * 0.09;
    WheelK { a: s * h.cos(), b: s * h.sin(), lum: (w.lum / 100.0) as f32 * 0.12 }
}

impl ColorOps {
    pub fn new(s: &DevelopSettings) -> ColorOps {
        let (mk, measured) = match Mixer::new(&s.mixer, s.process.process()) {
            Some(Mixer::Bands(k)) => (k, None),
            Some(Mixer::Measured(m)) => (MixerK::default(), Some(m)),
            None => (MixerK::new(&s.mixer), None),
        };
        let g = &s.grading;
        ColorOps {
            vibrance: (s.color.vibrance / 100.0) as f32,
            saturation: (s.color.saturation / 100.0) as f32,
            hue: mk.hue,
            sat: mk.sat,
            lum: mk.lum,
            mixer: !s.mixer.is_neutral() && measured.is_none(),
            measured,
            points: if crate::is_bw(s) { Vec::new() } else { point_ks(&s.point_colors) },
            bw: crate::is_bw(s).then(|| s.bw_mix.bands().map(|v| (v / 100.0) as f32)),
            grading: (!g.is_neutral()).then(|| {
                (
                    [wheel(&g.shadows), wheel(&g.midtones), wheel(&g.highlights), wheel(&g.global)],
                    (g.blending / 100.0) as f32,
                    (g.balance / 100.0) as f32,
                )
            }),
            primaries: PrimariesK::new(&s.primaries),
            hsl: if crate::is_bw(s) { None } else { HslK::new(&s.hsl_curves) },
            skin: oklch_hue_of_srgb_hue(25.0),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.vibrance == 0.0
            && self.saturation == 0.0
            && !self.mixer
            && self.measured.is_none()
            && self.points.is_empty()
            && self.bw.is_none()
            && self.grading.is_none()
            && self.primaries.is_none()
            && self.hsl.is_none()
    }

    /// `local_sat` (−1..1) and `local_hue` (radians) come from masks.
    #[inline]
    pub fn apply(&self, rgb: [f32; 3], local_sat: f32, local_hue: f32) -> [f32; 3] {
        if self.is_identity() && local_sat == 0.0 && local_hue == 0.0 {
            return rgb;
        }
        let rgb = match &self.measured {
            Some(m) => m.apply(rgb, 1.0),
            None => rgb,
        };
        let lab = oklab_from_2020(rgb);
        // only chroma changes (vibrance / saturation, maybe grading): scale a, b directly — the same
        // result as the OkLCh round trip without its sin / cos (and atan2 unless vibrance needs the hue)
        if !self.mixer && self.points.is_empty() && self.bw.is_none() && self.hsl.is_none() && local_hue == 0.0 {
            let mut lab = lab;
            let c0 = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
            let mut c = c0;
            if self.vibrance != 0.0 {
                let low = 1.0 - (c / 0.22).clamp(0.0, 1.0);
                let skin = if self.vibrance > 0.0 { 1.0 - 0.6 * (-(wrap(lab[2].atan2(lab[1]) - self.skin) / 0.35).powi(2)).exp() } else { 1.0 };
                c *= (1.0 + self.vibrance * low * low * skin * 1.2).max(0.0);
            }
            if self.saturation != 0.0 || local_sat != 0.0 {
                c *= (1.0 + self.saturation + local_sat).max(0.0);
            }
            if c0 > 0.0 {
                let k = c / c0;
                lab[1] *= k;
                lab[2] *= k;
            }
            return self.grade(lab);
        }
        let [mut l, mut c, mut h] = lab_to_lch(lab);
        if self.mixer {
            [l, c, h] = mixer_apply(&self.hue, &self.sat, &self.lum, [l, c, h], 1.0);
        }
        // the HSL curves: the hue picks the place on each curve, the mixer's own scales apply the
        // deltas (hue and luminance fading out towards neutrals, as the mixer's do)
        if let Some(hsl) = &self.hsl {
            let x = (h / TAU).rem_euclid(1.0);
            let [dh, ds, dl] = hsl.at(x);
            let chroma_w = (c / 0.12).min(1.0);
            h += dh * chroma_w;
            c *= (1.0 + ds).max(0.0);
            l += dl * chroma_w * l.max(0.05).sqrt();
        }
        for p in &self.points {
            [l, c, h] = p.apply([l, c, h]);
        }
        if self.vibrance != 0.0 {
            let low = 1.0 - (c / 0.22).clamp(0.0, 1.0);
            let skin = if self.vibrance > 0.0 { 1.0 - 0.6 * (-(wrap(h - self.skin) / 0.35).powi(2)).exp() } else { 1.0 };
            c *= (1.0 + self.vibrance * low * low * skin * 1.2).max(0.0);
        }
        if self.saturation != 0.0 || local_sat != 0.0 {
            c *= (1.0 + self.saturation + local_sat).max(0.0);
        }
        h += local_hue;
        if let Some(bw) = &self.bw {
            let w = band_weights(h);
            let mix: f32 = (0..8).map(|i| w[i] * bw[i]).sum();
            l = (l + mix * (c / 0.2).min(1.0) * 0.25).max(0.0);
            c = 0.0;
        }
        self.grade(lch_to_lab([l, c, h]))
    }

    /// Colour grading on OkLab, then back to linear Rec.2020.
    #[inline]
    fn grade(&self, mut lab: [f32; 3]) -> [f32; 3] {
        if let Some((wheels, blending, balance)) = &self.grading {
            let [ws, wm, wh] = grade_zones(lab[0], *blending, *balance);
            for (k, wt) in wheels.iter().zip([ws, wm, wh, 1.0]) {
                lab[1] += k.a * wt;
                lab[2] += k.b * wt;
                lab[0] += k.lum * wt;
            }
        }
        oklab_to_2020(lab)
    }
}

/// How much OkLab lightness `l` belongs to the shadows, midtones and highlights of colour grading
/// (`blending` 0..1, `balance` −1..1). The GPU mirrors it in `finish.wgsl`.
#[inline]
pub fn grade_zones(l: f32, blending: f32, balance: f32) -> [f32; 3] {
    let m = 0.5 - balance * 0.25;
    let width = 0.15 + blending * 0.5;
    let ws = 1.0 - smooth(m - width, m + width * 0.25, l);
    let wh = smooth(m - width * 0.25, m + width, l);
    [ws, (1.0 - ws - wh).max(0.0), wh]
}

/// A colourist wheel resolved for the per-pixel stage: the puck as a per-channel linear-Rec.2020
/// colour direction scaled by its saturation, and the master wheel as a signed amount.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrimaryK {
    pub rgb: [f32; 3],
    pub master: f32,
}

/// The Color wheels' reaches at a full wheel (±100): ASC CDL's SOP amounts. Lift's affine term
/// (blacks move by up to half the range, whites stay), gain's multiply (up to ×1.75), gamma's
/// power input (exponent 1/(1±p) ≈ 0.57..4 — midtones move hard), offset's add.
pub const PRIMARY_LIFT_REACH: f32 = 0.5;
pub const PRIMARY_GAIN_REACH: f32 = 0.75;
pub const PRIMARY_GAMMA_REACH: f32 = 0.75;
pub const PRIMARY_OFFSET_REACH: f32 = 0.1;

/// The Log wheels' base amounts (additive refinement over narrow zones): puck and master.
pub const PRIMARY_PUCK: f32 = 0.15;
pub const PRIMARY_MASTER: f32 = 0.15;
/// The Log highlight wheel, which multiplies and so starts at twice the additive base.
pub const PRIMARY_GAIN_MASTER: f32 = 0.3;

/// How much stronger each Log wheel reads — tuned by eye on rendered photos (the Color wheels
/// above now carry the CDL reaches): shadows halved (they stack on the Color lift), midtones as
/// tuned, the highlight ×3 — it *multiplies* its zone, so the full master pull lands at ×1.9 and
/// the full push at ×0.1 without ever clamping to black or leaping to white (×10 read as a
/// blow-out: ×4 at full throw, over after a few pixels of drag). (shadow, midtone, highlight,
/// offset.)
pub const LOG_STRENGTH: [f32; 4] = [0.5, 1.0, 3.0, 0.5];

/// The colourist primaries resolved once per render: the Color wheels (lift, gamma, gain, offset —
/// wide tonal overlap) and the Log wheels (shadow, midtone, highlight, offset — narrow ranges
/// between `low` and `high`); both sets apply, complimentary as Resolve's (see
/// [`lcc_develop::Primaries`]). `None` when every wheel is neutral. The GPU kernel reads the same
/// values ([`Self::words`]).
#[derive(Clone, Copy, Debug)]
pub struct PrimariesK {
    pub lift: PrimaryK,
    pub gamma: PrimaryK,
    pub gain: PrimaryK,
    pub offset: PrimaryK,
    pub shadow: PrimaryK,
    pub midtone: PrimaryK,
    pub highlight: PrimaryK,
    pub log_offset: PrimaryK,
    pub low: f32,
    pub high: f32,
}

/// f32s the GPU kernel reads for [`PrimariesK`]: the Color wheels then the Log wheels (rgb + master
/// each, in their orders), then `low` and `high`.
pub const PRIMARY_WORDS: usize = 34;

impl PrimariesK {
    pub fn new(p: &lcc_develop::Primaries) -> Option<PrimariesK> {
        if p.is_neutral() {
            return None;
        }
        let fl = LOG_STRENGTH;
        let log = &p.log_wheels;
        Some(PrimariesK {
            lift: primary_wheel(&p.lift, PRIMARY_LIFT_REACH, PRIMARY_LIFT_REACH),
            gamma: primary_wheel(&p.gamma, PRIMARY_GAMMA_REACH, PRIMARY_GAMMA_REACH),
            gain: primary_wheel(&p.gain, PRIMARY_GAIN_REACH, PRIMARY_GAIN_REACH),
            offset: primary_wheel(&p.offset, PRIMARY_OFFSET_REACH, PRIMARY_OFFSET_REACH),
            shadow: primary_wheel(&log.shadow, PRIMARY_PUCK * fl[0], PRIMARY_MASTER * fl[0]),
            midtone: primary_wheel(&log.midtone, PRIMARY_PUCK * fl[1], PRIMARY_MASTER * fl[1]),
            highlight: primary_wheel(&log.highlight, PRIMARY_PUCK * fl[2], PRIMARY_GAIN_MASTER * fl[2]),
            log_offset: primary_wheel(&log.offset, PRIMARY_PUCK * fl[3], PRIMARY_MASTER * fl[3]),
            low: p.low.clamp(0.0, 1.0) as f32,
            high: p.high.clamp(0.0, 1.0) as f32,
        })
    }

    pub fn words(&self) -> [f32; PRIMARY_WORDS] {
        let mut w = [0.0; PRIMARY_WORDS];
        let parts: [&[f32]; 16] = [
            &self.lift.rgb,
            std::slice::from_ref(&self.lift.master),
            &self.gamma.rgb,
            std::slice::from_ref(&self.gamma.master),
            &self.gain.rgb,
            std::slice::from_ref(&self.gain.master),
            &self.offset.rgb,
            std::slice::from_ref(&self.offset.master),
            &self.shadow.rgb,
            std::slice::from_ref(&self.shadow.master),
            &self.midtone.rgb,
            std::slice::from_ref(&self.midtone.master),
            &self.highlight.rgb,
            std::slice::from_ref(&self.highlight.master),
            &self.log_offset.rgb,
            std::slice::from_ref(&self.log_offset.master),
        ];
        let mut i = 0;
        for part in parts {
            for v in part {
                w[i] = *v;
                i += 1;
            }
        }
        w[32] = self.low;
        w[33] = self.high;
        w
    }

    /// The wheels on display-linear Rec.2020 `rgb` (after the tone map). The Color wheels are ASC
    /// CDL's SOP per channel — lift's affine `x·(1−l)+l` pivots at white, gain's multiply pivots
    /// at black, gamma is a power (exponent 1/(1+p)), offset adds last — so each carries its own
    /// tonal shape with no zone weights; then the Log wheels refine over their narrow ranges
    /// between `low` and `high` (both sets, complimentary). Sources: own implementation of the
    /// published ASC CDL SOP shape and the colorist community's documented lift/gamma/gain forms;
    /// wheel semantics read from Blackmagic's *Colorist Guide to DaVinci Resolve 18*.
    #[inline]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let t = lcc_color::luminance_2020(rgb).clamp(0.0, 1.0);
        let [ls, lm, lh] = log_zones(t, self.low, self.high);
        let mut out = rgb;
        for k in 0..3 {
            let l = self.lift.rgb[k] + self.lift.master;
            let g = self.gain.rgb[k] + self.gain.master;
            let p = self.gamma.rgb[k] + self.gamma.master;
            let mut x = rgb[k] * (1.0 - l) + l;
            x = (x * (1.0 + g)).max(0.0);
            let e = (1.0 / (1.0 + p)).clamp(0.25, 4.0);
            x = x.max(0.0).powf(e);
            x += self.offset.rgb[k] + self.offset.master;
            x += (self.shadow.rgb[k] + self.shadow.master) * ls;
            x += (self.midtone.rgb[k] + self.midtone.master) * lm;
            x *= 1.0 + (self.highlight.rgb[k] + self.highlight.master) * lh;
            x += self.log_offset.rgb[k] + self.log_offset.master;
            out[k] = x.max(0.0);
        }
        out
    }
}

/// A wheel's puck as a linear-Rec.2020 colour direction: the pure hue scaled by its saturation
/// (`puck_amt`), and the master wheel scaled by `master_amt` ([`PRIMARY_STRENGTH`] tunes both).
fn primary_wheel(w: &lcc_develop::Wheel, puck_amt: f32, master_amt: f32) -> PrimaryK {
    let c = hsv_to_rgb(w.hue.clamp(0.0, 360.0) as f32, 1.0, 1.0);
    let lin = c.map(lcc_color::transfer::srgb_to_linear);
    let v = SRGB.to_space(&REC2020).apply_f32(lin);
    let s = (w.sat.clamp(0.0, 100.0) / 100.0) as f32 * puck_amt;
    PrimaryK { rgb: v.map(|k| k * s), master: (w.lum.clamp(-100.0, 100.0) / 100.0) as f32 * master_amt }
}

/// The Log wheels' zones: narrowly separated ranges with their boundaries at `low` and `high`
/// (0..1). The GPU mirrors it in `finish.wgsl`.
#[inline]
pub fn log_zones(t: f32, low: f32, high: f32) -> [f32; 3] {
    let low = low.clamp(0.02, 0.9);
    let hm = high.clamp(low + 0.05, 0.98);
    let ws = 1.0 - smooth(low - 0.10, low + 0.10, t);
    let wh = smooth(hm - 0.10, hm + 0.10, t);
    [ws, (1.0 - ws - wh).max(0.0), wh]
}

/// Number of per-mask colour-grading terms: (a, b, lightness) of the shadows, midtones,
/// highlights and global wheels.
pub const LOCAL_GRADE_TERMS: usize = 12;

/// A mask's colour-grading wheels as alpha-summable terms (see [`LOCAL_GRADE_TERMS`]).
pub fn local_grade_terms(a: &lcc_develop::LocalAdjustments) -> [f32; LOCAL_GRADE_TERMS] {
    let mut t = [0.0; LOCAL_GRADE_TERMS];
    for (i, w) in a.grading_wheels().iter().enumerate() {
        let k = wheel(w);
        t[i * 3] = k.a;
        t[i * 3 + 1] = k.b;
        t[i * 3 + 2] = k.lum;
    }
    t
}

/// Colour grading inside masks: the alpha-weighted sum `g` of the masks' [`local_grade_terms`]
/// applied to linear Rec.2020 `rgb`, zones as the Color Grading panel's at its default blending (50)
/// and balance (0). Sources: own design (the global grading's zones and wheels, weighted per mask).
#[inline]
pub fn local_grade(rgb: [f32; 3], g: &[f32; LOCAL_GRADE_TERMS]) -> [f32; 3] {
    if g.iter().all(|v| *v == 0.0) {
        return rgb;
    }
    let mut lab = oklab_from_2020(rgb);
    let [ws, wm, wh] = grade_zones(lab[0], 0.5, 0.0);
    for (i, wt) in [ws, wm, wh, 1.0].into_iter().enumerate() {
        lab[1] += g[i * 3] * wt;
        lab[2] += g[i * 3 + 1] * wt;
        lab[0] += g[i * 3 + 2] * wt;
    }
    oklab_to_2020(lab)
}

/// Hue rotation (OkLCh radians) of a calibration primary at ±100.
pub const CALIB_HUE: f32 = 0.5;
/// Chroma scale of a calibration primary at ±100 (`1 ± CALIB_SAT`).
pub const CALIB_SAT: f32 = 0.6;

/// Hue rotation (degrees in CIE xy around the white point) of a ProPhoto primary at ±100 (process V9).
pub const CALIB_XY_HUE: f64 = 30.0;
/// Saturation ±100 (process V9) scales a ProPhoto primary's xy distance from white by `1 + a·v + b·v²`
/// (`v` = slider / 100), per primary (red, green, blue): fitted to Lightroom Classic's round-17 charts.
pub const CALIB_XY_SAT: [(f64, f64); 3] = [(1.367, 0.732), (1.656, 0.959), (0.96, 0.329)];

/// The calibration panel's primaries as a white-preserving 3×3 matrix on linear Rec.2020 (row-major),
/// `None` when neutral. V1–V8: each Rec.2020 primary is rotated in OkLCh hue and scaled in chroma at
/// constant OkLab lightness. V9 follows Lightroom Classic ([`calibration_matrix_xy`]). Both rescale the
/// columns so that neutral (1, 1, 1) maps to itself.
pub fn calibration_matrix(c: &Calibration, process: lcc_develop::Process) -> Option<[[f32; 3]; 3]> {
    let prim = c.primaries();
    if prim.iter().all(|(h, s)| *h == 0.0 && *s == 0.0) {
        return None;
    }
    if process >= lcc_develop::Process::V9 {
        return calibration_matrix_xy(&prim);
    }
    let mut p = [[0.0f64; 3]; 3];
    for (i, (hue, sat)) in prim.iter().enumerate() {
        let mut e = [0.0f32; 3];
        e[i] = 1.0;
        let [l, ch, h] = lab_to_lch(oklab_from_2020(e));
        let h2 = h + (hue.clamp(-100.0, 100.0) / 100.0) as f32 * CALIB_HUE;
        let c2 = ch * (1.0 + (sat.clamp(-100.0, 100.0) / 100.0) as f32 * CALIB_SAT);
        let q = oklab_to_2020(lch_to_lab([l, c2, h2]));
        for (r, row) in p.iter_mut().enumerate() {
            row[i] = q[r] as f64;
        }
    }
    let m = Mat3(p);
    let k = m.inverse()?.apply([1.0, 1.0, 1.0]);
    Some(std::array::from_fn(|r| std::array::from_fn(|col| (p[r][col] * k[col]) as f32)))
}

/// Process V9 calibration, as measured on Lightroom Classic (round 17 charts): each ProPhoto primary,
/// adapted to D65, moves in CIE xy around the white point: hue turns it by up to ±[`CALIB_XY_HUE`]°,
/// saturation scales its distance from white ([`CALIB_XY_SAT`]). The moved primaries, rescaled to keep
/// white, replace the old ones: the matrix maps a colour's ProPhoto coordinates onto them.
fn calibration_matrix_xy(prim: &[(f64, f64); 3]) -> Option<[[f32; 3]; 3]> {
    let to_xyz = bradford(D50, D65).mul(&PROPHOTO.to_xyz());
    let from_xyz = to_xyz.inverse()?;
    let white = to_xyz.apply([1.0, 1.0, 1.0]);
    let w = Xy::from_xyz(white);
    let mut p = [[0.0f64; 3]; 3];
    for (i, (hue, sat)) in prim.iter().enumerate() {
        let col = Xy::from_xyz([to_xyz.0[0][i], to_xyz.0[1][i], to_xyz.0[2][i]]);
        let (dx, dy) = (col.x - w.x, col.y - w.y);
        let a = (hue.clamp(-100.0, 100.0) / 100.0 * CALIB_XY_HUE).to_radians();
        let v = sat.clamp(-100.0, 100.0) / 100.0;
        let (ka, kb) = CALIB_XY_SAT.get(i).copied().unwrap_or((0.0, 0.0));
        let scale = (1.0 + ka * v + kb * v * v).max(0.0);
        let (rx, ry) = ((dx * a.cos() - dy * a.sin()) * scale, (dx * a.sin() + dy * a.cos()) * scale);
        let (x, y) = (w.x + rx, w.y + ry);
        // A primary moved past the spectral locus can land at y ≤ 0 (ProPhoto blue sits at y ≈ 0.0001): it's still a
        // direction in XYZ, and the white-keeping column scale below fixes its sign. Only y ≈ 0 has no direction.
        if !(y.abs() > 1e-6 && x.is_finite()) {
            return None;
        }
        let q = Xy::new(x, y).to_xyz();
        for (r, row) in p.iter_mut().enumerate() {
            row[i] = q[r];
        }
    }
    let m = Mat3(p);
    let k = m.inverse()?.apply(white);
    let moved = Mat3(std::array::from_fn(|r| std::array::from_fn(|c| p[r][c] * k[c])));
    let rec = REC2020.from_xyz().mul(&moved).mul(&from_xyz).mul(&REC2020.to_xyz());
    let out = rec.to_f32();
    out.iter().flatten().all(|v| v.is_finite()).then_some(out)
}

/// Shadows-tint strength at ±100: the green channel's relative change in deep shadows.
pub const SHADOW_TINT: f32 = 0.3;

/// Calibration in scene-linear light (before tone mapping): the primaries matrix, then the shadows
/// tint (green ↔ magenta, weighted towards dark tones, luminance kept).
#[inline]
pub fn calibrate(c: [f32; 3], m: Option<&[[f32; 3]; 3]>, shadow_tint: f32) -> [f32; 3] {
    let mut c = c;
    if let Some(m) = m {
        c = std::array::from_fn(|r| (m[r][0] * c[0] + m[r][1] * c[1] + m[r][2] * c[2]).max(0.0));
    }
    if shadow_tint != 0.0 {
        let y0 = lcc_color::luminance_2020(c);
        let w = 1.0 - smooth(-5.0, -0.5, (y0.max(1e-7) / 0.18).log2());
        c[1] *= (1.0 - SHADOW_TINT * shadow_tint * w).max(0.0);
        let y1 = lcc_color::luminance_2020(c).max(1e-9);
        c = c.map(|v| v * y0 / y1);
    }
    c
}

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcc_color::perceptual::lab_to_lch;

    #[test]
    fn weights_partition_unity() {
        for i in 0..360 {
            let h = (i as f32).to_radians() - PI;
            let w = band_weights(h);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "{i}: {s}");
        }
        // at a band centre, that band dominates
        let w = band_weights(band_hues()[5]);
        assert!(w[5] > 0.99);
    }

    #[test]
    fn neutral_is_identity() {
        let ops = ColorOps::new(&DevelopSettings::default());
        assert!(ops.is_identity());
        assert_eq!(ops.apply([0.2, 0.3, 0.4], 0.0, 0.0), [0.2, 0.3, 0.4]);
    }

    #[test]
    fn saturation_and_bw() {
        let mut s = DevelopSettings::default();
        s.color.saturation = -100.0;
        let g = ColorOps::new(&s).apply([0.4, 0.1, 0.05], 0.0, 0.0);
        assert!((g[0] - g[1]).abs() < 1e-3 && (g[1] - g[2]).abs() < 1e-3, "{g:?}");
        let mut s = DevelopSettings { treatment: lcc_develop::Treatment::Bw, ..Default::default() };
        let ops = ColorOps::new(&s);
        let b = ops.apply([0.05, 0.1, 0.5], 0.0, 0.0);
        assert!((b[0] - b[2]).abs() < 1e-3);
        // raising blue in the B&W mix brightens blue things
        s.bw_mix.blue = 80.0;
        let b2 = ColorOps::new(&s).apply([0.05, 0.1, 0.5], 0.0, 0.0);
        assert!(b2[1] > b[1]);
    }

    #[test]
    fn mixer_targets_its_band() {
        // V8 greys the band out; V9, as Lightroom Classic, leaves a vivid blue a trace of colour
        for (process, most) in [(lcc_develop::Process::V8, 0.02), (lcc_develop::Process::V9, 0.04)] {
            let mut s = DevelopSettings { process: process.version(), ..Default::default() };
            s.mixer.blue.sat = -100.0;
            let ops = ColorOps::new(&s);
            let blue = [0.02, 0.05, 0.4];
            let red = [0.4, 0.03, 0.02];
            let cb0 = lab_to_lch(oklab_from_2020(blue))[1];
            let cb = lab_to_lch(oklab_from_2020(ops.apply(blue, 0.0, 0.0)))[1];
            let cr0 = lab_to_lch(oklab_from_2020(red))[1];
            let cr = lab_to_lch(oklab_from_2020(ops.apply(red, 0.0, 0.0)))[1];
            assert!(cb < most && cb < 0.2 * cb0, "{process:?}: {cb} of {cb0}");
            assert!((cr - cr0).abs() < 0.01, "{process:?}");
        }
    }

    /// Settings with one Color Mixer slider (`band`, `kind` 0 hue / 1 saturation / 2 luminance) at `v` (−1 … 1).
    fn one_slider(process: lcc_develop::Process, band: usize, kind: usize, v: f32) -> DevelopSettings {
        let mut s = DevelopSettings { process: process.version(), ..Default::default() };
        let b = [
            &mut s.mixer.red,
            &mut s.mixer.orange,
            &mut s.mixer.yellow,
            &mut s.mixer.green,
            &mut s.mixer.aqua,
            &mut s.mixer.blue,
            &mut s.mixer.purple,
            &mut s.mixer.magenta,
        ]
        .into_iter()
        .nth(band)
        .expect("8 bands");
        *[&mut b.hue, &mut b.sat, &mut b.lum].into_iter().nth(kind).expect("3 kinds") = f64::from(v) * 100.0;
        s
    }

    /// Process V9's mixer against Lightroom Classic's own exports of our round-12 colour charts (one slider at a time,
    /// 8 Nov 2026; patches read through the files' embedded profiles): the chart's baseline colour in, and what
    /// Lightroom Classic made of it at that slider out. The measured tables
    /// ([`crate::mixer_measured`]) keep every patch close and the average hue within about 2° (8 Nov 2026: 2.3°, chroma
    /// 3 %, L 0.003), where V8 missed by 13°.
    #[test]
    fn v9_mixer_follows_lightroom_classic_on_our_charts() {
        // (band, kind 0 hue / 1 sat / 2 lum, slider, linear Rec.2020 in, Lightroom Classic's OkLCh out (L, C, hue °))
        const LRC: [(usize, usize, f32, [f32; 3], [f32; 3]); 48] = [
            (2, 0, -1.00, [0.10981, 0.08999, 0.03370], [0.4114, 0.0722, 39.77]),
            (2, 0, -1.00, [0.11301, 0.08761, 0.01165], [0.3979, 0.1039, 47.35]),
            (2, 0, -1.00, [0.11947, 0.08728, 0.00955], [0.3954, 0.1083, 48.03]),
            (2, 0, -1.00, [0.27291, 0.23852, 0.11997], [0.5816, 0.0707, 38.54]),
            (2, 0, -1.00, [0.29927, 0.23459, 0.03210], [0.5520, 0.1412, 47.00]),
            (2, 0, -1.00, [0.30016, 0.22667, 0.02902], [0.5514, 0.1481, 47.91]),
            (2, 0, -0.50, [0.10981, 0.08999, 0.03370], [0.4347, 0.0603, 63.88]),
            (2, 0, -0.50, [0.11301, 0.08761, 0.01165], [0.4297, 0.0912, 68.23]),
            (2, 0, -0.50, [0.11947, 0.08728, 0.00955], [0.4279, 0.0940, 71.86]),
            (2, 0, -0.50, [0.27291, 0.23852, 0.11997], [0.6056, 0.0610, 61.80]),
            (2, 0, -0.50, [0.29927, 0.23459, 0.03210], [0.5929, 0.1235, 69.30]),
            (2, 0, -0.50, [0.30016, 0.22667, 0.02902], [0.5922, 0.1293, 68.70]),
            (2, 0, 0.50, [0.09473, 0.09614, 0.03343], [0.4445, 0.0720, 122.51]),
            (2, 0, 0.50, [0.10083, 0.09354, 0.01140], [0.4433, 0.1100, 121.44]),
            (2, 0, 0.50, [0.10346, 0.09602, 0.01243], [0.4434, 0.1082, 121.59]),
            (2, 0, 0.50, [0.24460, 0.24815, 0.12538], [0.6127, 0.0719, 123.48]),
            (2, 0, 0.50, [0.26502, 0.25261, 0.04037], [0.6102, 0.1427, 122.84]),
            (2, 0, 0.50, [0.26697, 0.25276, 0.02632], [0.6109, 0.1556, 120.53]),
            (3, 0, -1.00, [0.07572, 0.10514, 0.04736], [0.4597, 0.0424, 93.79]),
            (3, 0, -1.00, [0.06401, 0.11768, 0.01636], [0.4715, 0.0801, 96.19]),
            (3, 0, -1.00, [0.06401, 0.11768, 0.01636], [0.4714, 0.0839, 96.63]),
            (3, 0, -1.00, [0.20929, 0.26617, 0.15549], [0.6301, 0.0398, 89.68]),
            (3, 0, -1.00, [0.18517, 0.29013, 0.08527], [0.6411, 0.0812, 95.64]),
            (3, 0, -1.00, [0.16730, 0.30506, 0.04200], [0.6505, 0.1140, 96.98]),
            (1, 0, -1.00, [0.12541, 0.08133, 0.03745], [0.4120, 0.0926, 17.95]),
            (1, 0, -1.00, [0.14980, 0.07063, 0.00925], [0.3817, 0.1603, 15.81]),
            (1, 0, -1.00, [0.14952, 0.07061, 0.01130], [0.3817, 0.1603, 15.81]),
            (1, 0, -1.00, [0.30022, 0.21548, 0.13266], [0.5819, 0.0819, 21.59]),
            (1, 0, -1.00, [0.36294, 0.19210, 0.04868], [0.5390, 0.1835, 17.67]),
            (1, 0, -1.00, [0.39170, 0.18339, 0.03087], [0.5287, 0.2177, 15.83]),
            (0, 0, 0.50, [0.13438, 0.07038, 0.07182], [0.4585, 0.0501, 28.27]),
            (0, 0, 0.50, [0.17724, 0.04808, 0.05168], [0.4724, 0.0940, 32.60]),
            (0, 0, 0.50, [0.21494, 0.02863, 0.03794], [0.4800, 0.1381, 32.38]),
            (0, 0, 0.50, [0.31659, 0.19950, 0.19902], [0.6314, 0.0500, 30.73]),
            (0, 0, 0.50, [0.39634, 0.15937, 0.15996], [0.6422, 0.0984, 30.57]),
            (0, 0, 0.50, [0.46809, 0.12232, 0.13660], [0.6515, 0.1395, 31.70]),
            (2, 1, -0.50, [0.10981, 0.08999, 0.03370], [0.4254, 0.0299, 84.56]),
            (2, 1, -0.50, [0.11301, 0.08761, 0.01165], [0.4168, 0.0525, 80.96]),
            (2, 1, -0.50, [0.11947, 0.08728, 0.00955], [0.4168, 0.0525, 80.96]),
            (2, 1, -0.50, [0.27291, 0.23852, 0.11997], [0.5967, 0.0321, 91.93]),
            (2, 1, -0.50, [0.29927, 0.23459, 0.03210], [0.5747, 0.0677, 85.29]),
            (2, 1, -0.50, [0.30016, 0.22667, 0.02902], [0.5767, 0.0704, 88.20]),
            (2, 2, -0.50, [0.10981, 0.08999, 0.03370], [0.4077, 0.0530, 92.05]),
            (2, 2, -0.50, [0.11301, 0.08761, 0.01165], [0.3972, 0.0796, 91.83]),
            (2, 2, -0.50, [0.11947, 0.08728, 0.00955], [0.3972, 0.0796, 91.83]),
            (2, 2, -0.50, [0.27291, 0.23852, 0.11997], [0.5414, 0.0563, 87.07]),
            (2, 2, -0.50, [0.29927, 0.23459, 0.03210], [0.5007, 0.1016, 90.22]),
            (2, 2, -0.50, [0.30016, 0.22667, 0.02902], [0.5006, 0.1047, 90.74]),
        ];
        let err = |process: lcc_develop::Process| {
            let (mut sum, mut worst) = ([0.0f32; 3], [0.0f32; 3]);
            for (band, kind, v, rgb, want) in LRC {
                let out = lab_to_lch(oklab_from_2020(ColorOps::new(&one_slider(process, band, kind, v)).apply(rgb, 0.0, 0.0)));
                let dh = ((out[2].to_degrees() - want[2] + 540.0).rem_euclid(360.0) - 180.0).abs();
                let e = [dh, (out[1] / want[1] - 1.0).abs(), (out[0] - want[0]).abs()];
                for i in 0..3 {
                    sum[i] += e[i] / LRC.len() as f32;
                    worst[i] = worst[i].max(e[i]);
                }
            }
            (sum, worst)
        };
        let (v9, v9_worst) = err(lcc_develop::Process::V9);
        let (v8, _) = err(lcc_develop::Process::V8);
        let msg = format!("V9 mean hue {:.2}° chroma {:.3} L {:.4}, worst {v9_worst:?}; V8 mean {v8:?}", v9[0], v9[1], v9[2]);
        assert!(v9_worst[0] < 7.0 && v9_worst[1] < 0.15 && v9_worst[2] < 0.012, "{msg}");
        assert!(v9[0] < 3.0 && v9[1] < 0.04 && v9[2] < 0.004, "{msg}");
        assert!(v9[0] < 0.5 * v8[0] && v9[1] < v8[1] && v9[2] < v8[2], "{msg}");
    }

    #[test]
    fn older_processes_keep_the_band_mixer_and_v9_neutrals_stay() {
        let mut m = lcc_develop::Mixer::default();
        m.yellow.hue = -60.0;
        m.green.sat = 30.0;
        m.blue.lum = -40.0;
        for p in [lcc_develop::Process::V8, lcc_develop::Process::V1] {
            assert_eq!(Mixer::new(&m, p), Some(Mixer::Bands(MixerK::new(&m))), "{p:?}");
        }
        assert!(matches!(Mixer::new(&m, lcc_develop::Process::V9), Some(Mixer::Measured(_))));
        assert_eq!(Mixer::new(&lcc_develop::Mixer::default(), lcc_develop::Process::V9), None);
        let s = one_slider(lcc_develop::Process::V9, 2, 0, -1.0);
        let ops = ColorOps::new(&s);
        assert!(!ops.mixer && ops.measured.is_some() && !ops.is_identity());
        for g in [0.0, 0.18, 1.0] {
            let o = ops.apply([g; 3], 0.0, 0.0);
            assert!(o.iter().all(|v| (v - g).abs() < 1e-5), "grey {g} → {o:?}");
        }
        let v8 = ColorOps::new(&one_slider(lcc_develop::Process::V8, 2, 0, -1.0));
        assert!(v8.mixer && v8.measured.is_none());
    }

    #[test]
    fn calibration_keeps_white_and_moves_primaries() {
        assert!(calibration_matrix(&Calibration::default(), lcc_develop::Process::V8).is_none());
        let cal = Calibration { red_hue: 60.0, blue_sat: -50.0, ..Default::default() };
        let m = calibration_matrix(&cal, lcc_develop::Process::V8).unwrap();
        let grey = calibrate([0.3, 0.3, 0.3], Some(&m), 0.0);
        assert!(grey.iter().all(|v| (v - 0.3).abs() < 1e-4), "{grey:?}");
        // red rotates in hue, blue loses chroma
        let red = [0.4, 0.05, 0.03];
        let h0 = lab_to_lch(oklab_from_2020(red))[2];
        let h1 = lab_to_lch(oklab_from_2020(calibrate(red, Some(&m), 0.0)))[2];
        assert!(h1 - h0 > 0.05, "{h0} -> {h1}");
        let blue = [0.03, 0.05, 0.4];
        let c0 = lab_to_lch(oklab_from_2020(blue))[1];
        let c1 = lab_to_lch(oklab_from_2020(calibrate(blue, Some(&m), 0.0)))[1];
        assert!(c1 < c0 * 0.97, "{c0} -> {c1}");
    }

    #[rustfmt::skip]
    // Generated by r17calconst.py from Lightroom Classic's round-17 chart exports (16-bit ProPhoto, read to OkLab).
    const R17_BASE: [[f32; 3]; 24] = [[0.24970, 0.03479, -0.02004], [0.24961, -0.00013, -0.11015], [0.24983, 0.05505, -0.09526], [0.34985, 0.10999, 0.00001], [0.34979, 0.09531, 0.05505], [0.34980, -0.00001, -0.11004], [0.34984, 0.09528, -0.05501], [0.44986, 0.05502, -0.09526], [0.54988, 0.09522, 0.05502], [0.64992, 0.05494, 0.09528], [0.64988, 0.00001, -0.11004], [0.74994, 0.10998, 0.00001], [0.74994, 0.05494, 0.09527], [0.74992, -0.11006, -0.00001], [0.74990, -0.05504, -0.09531], [0.74992, 0.00000, -0.11004], [0.84997, -0.00008, 0.11001], [0.84996, -0.05509, 0.09527], [0.84995, -0.09535, 0.05499], [0.84994, -0.09531, -0.05503], [0.89999, -0.00008, 0.11001], [0.89999, -0.05509, 0.09527], [0.89999, -0.09536, 0.05501], [0.89997, -0.11009, -0.00001]];
    /// (primary 0 red / 1 green / 2 blue, hue slider, sat slider, LrC OkLab of every patch)
    const R17_CAL: [(usize, f64, f64, [[f32; 3]; 24]); 12] = [
        (
            0,
            -100.0,
            0.0,
            [
                [0.24212, 0.04636, -0.02351],
                [0.24153, 0.02637, -0.08750],
                [0.23857, 0.08162, -0.08192],
                [0.32631, 0.15469, -0.03723],
                [0.31982, 0.16407, -0.01150],
                [0.34312, 0.01722, -0.08434],
                [0.32937, 0.12997, -0.06490],
                [0.43524, 0.07843, -0.08306],
                [0.52179, 0.15351, -0.00458],
                [0.63070, 0.09883, 0.03222],
                [0.64612, 0.00662, -0.08701],
                [0.71191, 0.17853, -0.04210],
                [0.73732, 0.08585, 0.02982],
                [0.76926, -0.13164, 0.02955],
                [0.76354, -0.07925, -0.06178],
                [0.74684, 0.00471, -0.09089],
                [0.84897, 0.01445, 0.05306],
                [0.85750, -0.05516, 0.06403],
                [0.86458, -0.11136, 0.05813],
                [0.87669, -0.13788, -0.01294],
                [0.89977, 0.01327, 0.05004],
                [0.90686, -0.05254, 0.06083],
                [0.91267, -0.10825, 0.05642],
                [0.91447, -0.12158, 0.02743],
            ],
        ),
        (
            0,
            100.0,
            0.0,
            [
                [0.25741, 0.02323, -0.01657],
                [0.25739, -0.03129, -0.12902],
                [0.26266, 0.02381, -0.10475],
                [0.37667, 0.05884, 0.03866],
                [0.38156, 0.02987, 0.09449],
                [0.35634, -0.02056, -0.13244],
                [0.37099, 0.06130, -0.04488],
                [0.46388, 0.03089, -0.10526],
                [0.57662, 0.03800, 0.11014],
                [0.66844, 0.00838, 0.15567],
                [0.65318, -0.00627, -0.12734],
                [0.77697, 0.06435, 0.03576],
                [0.76678, 0.00917, 0.16899],
                [0.72455, -0.07842, -0.03260],
                [0.73651, -0.03883, -0.12106],
                [0.75243, -0.00382, -0.12275],
                [0.85141, -0.02625, 0.18467],
                [0.84202, -0.05575, 0.13796],
                [0.83234, -0.07444, 0.05085],
                [0.82828, -0.05545, -0.08175],
                [0.90036, -0.03032, 0.20265],
                [0.89272, -0.05988, 0.14803],
                [0.88389, -0.07698, 0.05271],
                [0.87507, -0.07763, -0.03306],
            ],
        ),
        (
            0,
            0.0,
            -100.0,
            [
                [0.25879, 0.02041, -0.01193],
                [0.25252, 0.00761, -0.07231],
                [0.26107, 0.03446, -0.05994],
                [0.38239, 0.06058, 0.00006],
                [0.38118, 0.05824, 0.02613],
                [0.35131, 0.00575, -0.07137],
                [0.37448, 0.05381, -0.03202],
                [0.46293, 0.03368, -0.05848],
                [0.58234, 0.05658, 0.02813],
                [0.67410, 0.03540, 0.04789],
                [0.64961, 0.00299, -0.07816],
                [0.78795, 0.05579, 0.00002],
                [0.77576, 0.03149, 0.04314],
                [0.71942, -0.07566, 0.00000],
                [0.73108, -0.04023, -0.07573],
                [0.74966, 0.00233, -0.08327],
                [0.85754, 0.00315, 0.04932],
                [0.84348, -0.02916, 0.05027],
                [0.82938, -0.06473, 0.03697],
                [0.82605, -0.07350, -0.04398],
                [0.90750, 0.00292, 0.04600],
                [0.89486, -0.02798, 0.04826],
                [0.88165, -0.06605, 0.03776],
                [0.87400, -0.08129, -0.00000],
            ],
        ),
        (
            0,
            0.0,
            100.0,
            [
                [0.24107, 0.04930, -0.02798],
                [0.24672, -0.01700, -0.14252],
                [0.24105, 0.07338, -0.12635],
                [0.31742, 0.16343, -0.00018],
                [0.32865, 0.12079, 0.07388],
                [0.34837, -0.01219, -0.14396],
                [0.32704, 0.13808, -0.07738],
                [0.43663, 0.07509, -0.12959],
                [0.51752, 0.12942, 0.09398],
                [0.62835, 0.06184, 0.15467],
                [0.64970, -0.00396, -0.13330],
                [0.69954, 0.18658, -0.00012],
                [0.72460, 0.06575, 0.16836],
                [0.77461, -0.13646, -0.00004],
                [0.76512, -0.06342, -0.10572],
                [0.74956, -0.00239, -0.12706],
                [0.84209, -0.01603, 0.19601],
                [0.85517, -0.08880, 0.15403],
                [0.86561, -0.12013, 0.06985],
                [0.86624, -0.10697, -0.06073],
                [0.89128, -0.02058, 0.21737],
                [0.90313, -0.09493, 0.16474],
                [0.91255, -0.11828, 0.06871],
                [0.91810, -0.12881, -0.00003],
            ],
        ),
        (
            1,
            -100.0,
            0.0,
            [
                [0.24498, 0.02524, -0.03088],
                [0.24862, -0.04440, -0.13722],
                [0.24225, 0.01052, -0.12554],
                [0.33123, 0.08175, -0.01460],
                [0.32941, 0.06327, 0.05713],
                [0.35011, -0.02579, -0.13844],
                [0.33582, 0.06863, -0.08344],
                [0.44327, 0.02636, -0.12591],
                [0.53153, 0.06285, 0.05969],
                [0.63939, 0.02381, 0.12289],
                [0.65032, -0.00693, -0.12955],
                [0.73473, 0.08568, -0.01506],
                [0.74226, 0.02494, 0.13223],
                [0.76670, -0.06529, 0.01121],
                [0.76572, -0.03130, -0.10030],
                [0.75002, -0.00419, -0.12442],
                [0.84654, -0.02276, 0.17421],
                [0.85379, -0.06155, 0.16354],
                [0.86164, -0.08092, 0.10212],
                [0.87505, -0.04573, -0.04847],
                [0.89626, -0.02624, 0.19054],
                [0.90327, -0.06797, 0.17974],
                [0.91094, -0.08767, 0.11089],
                [0.91901, -0.05794, 0.00997],
            ],
        ),
        (
            1,
            100.0,
            0.0,
            [
                [0.25418, 0.04390, -0.01003],
                [0.25050, 0.02895, -0.07790],
                [0.25714, 0.07645, -0.05991],
                [0.36790, 0.13193, 0.01478],
                [0.36862, 0.12321, 0.05217],
                [0.34947, 0.01855, -0.07769],
                [0.36184, 0.11714, -0.02846],
                [0.45595, 0.07396, -0.06030],
                [0.56583, 0.12343, 0.04972],
                [0.65931, 0.08167, 0.06871],
                [0.64900, 0.00690, -0.08448],
                [0.76196, 0.12921, 0.01278],
                [0.75929, 0.07526, 0.06237],
                [0.72959, -0.16800, -0.01322],
                [0.73274, -0.09017, -0.08611],
                [0.74926, 0.00490, -0.08899],
                [0.85372, 0.01296, 0.06094],
                [0.84638, -0.05316, 0.04665],
                [0.83623, -0.13412, 0.02649],
                [0.81782, -0.18281, -0.06219],
                [0.90379, 0.01204, 0.05740],
                [0.89697, -0.05015, 0.04387],
                [0.88554, -0.13728, 0.02721],
                [0.86939, -0.19812, -0.01562],
            ],
        ),
        (
            1,
            0.0,
            -100.0,
            [
                [0.24222, 0.02150, -0.01258],
                [0.24082, 0.00670, -0.07498],
                [0.23365, 0.03750, -0.06521],
                [0.32595, 0.07374, 0.00004],
                [0.32257, 0.06820, 0.03407],
                [0.34307, 0.00539, -0.07315],
                [0.32774, 0.06348, -0.03729],
                [0.43587, 0.03610, -0.06267],
                [0.52749, 0.06385, 0.03314],
                [0.63782, 0.03924, 0.05599],
                [0.64598, 0.00288, -0.07928],
                [0.72999, 0.07708, 0.00002],
                [0.74414, 0.03665, 0.05271],
                [0.77057, -0.05673, 0.00001],
                [0.76716, -0.03148, -0.06380],
                [0.74664, 0.00225, -0.08444],
                [0.85113, 0.00315, 0.05224],
                [0.85883, -0.02551, 0.04398],
                [0.87012, -0.04674, 0.02651],
                [0.88092, -0.04703, -0.02986],
                [0.90197, 0.00293, 0.04903],
                [0.90889, -0.02397, 0.04129],
                [0.92137, -0.04412, 0.02502],
                [0.92275, -0.04988, 0.00001],
            ],
        ),
        (
            1,
            0.0,
            100.0,
            [
                [0.25720, 0.04711, -0.02687],
                [0.25800, -0.01128, -0.13862],
                [0.26525, 0.06890, -0.11891],
                [0.37197, 0.13990, -0.00006],
                [0.37543, 0.11481, 0.07401],
                [0.35639, -0.00978, -0.14133],
                [0.37060, 0.12181, -0.06951],
                [0.46311, 0.07028, -0.12147],
                [0.56971, 0.11921, 0.07551],
                [0.66145, 0.06355, 0.13575],
                [0.65325, -0.00349, -0.13166],
                [0.76632, 0.13405, -0.00001],
                [0.75503, 0.06582, 0.14167],
                [0.72517, -0.18154, -0.00013],
                [0.72825, -0.08222, -0.12326],
                [0.75253, -0.00207, -0.12549],
                [0.84856, -0.01375, 0.18765],
                [0.83990, -0.10093, 0.17532],
                [0.82235, -0.16990, 0.10094],
                [0.80043, -0.17862, -0.08887],
                [0.89695, -0.01767, 0.20696],
                [0.88871, -0.11227, 0.19515],
                [0.86639, -0.18658, 0.11141],
                [0.86344, -0.22097, -0.00019],
            ],
        ),
        (
            2,
            -100.0,
            0.0,
            [
                [0.25796, 0.01789, -0.01427],
                [0.28568, -0.15791, -0.07491],
                [0.27527, -0.07819, -0.06984],
                [0.37006, 0.11540, 0.01043],
                [0.36778, 0.13641, 0.05856],
                [0.37844, -0.15346, -0.08071],
                [0.37246, 0.04511, -0.03834],
                [0.47493, -0.05789, -0.07226],
                [0.56185, 0.14397, 0.05677],
                [0.64548, 0.13189, 0.08797],
                [0.65593, -0.16538, -0.09796],
                [0.76823, 0.11082, 0.00903],
                [0.73885, 0.12938, 0.08245],
                [0.72894, -0.14375, -0.00969],
                [0.71296, -0.28997, -0.10810],
                [0.72916, -0.20170, -0.11454],
                [0.82567, 0.08658, 0.08716],
                [0.83226, 0.02537, 0.08200],
                [0.83239, -0.06140, 0.04452],
                [0.79965, -0.28963, -0.07547],
                [0.87298, 0.08477, 0.08341],
                [0.88286, 0.02611, 0.08225],
                [0.88454, -0.06311, 0.04592],
                [0.87401, -0.16326, -0.01108],
            ],
        ),
        (
            2,
            100.0,
            0.0,
            [
                [0.24203, 0.05142, -0.02560],
                [0.26142, 0.14181, -0.10988],
                [0.25491, 0.14513, -0.09622],
                [0.32838, 0.10268, -0.01111],
                [0.33067, 0.04682, 0.04964],
                [0.33966, 0.15409, -0.12589],
                [0.32978, 0.14451, -0.07057],
                [0.43070, 0.16527, -0.11397],
                [0.53498, 0.03900, 0.05142],
                [0.64967, -0.02960, 0.09999],
                [0.60873, 0.16520, -0.14971],
                [0.72880, 0.10841, -0.01123],
                [0.75624, -0.03406, 0.10508],
                [0.76827, -0.08322, 0.00804],
                [0.74091, 0.07490, -0.11020],
                [0.67057, 0.19632, -0.17828],
                [0.85306, -0.08970, 0.11383],
                [0.85396, -0.11805, 0.09702],
                [0.86356, -0.12247, 0.06354],
                [0.87739, -0.01001, -0.04229],
                [0.90126, -0.08855, 0.11131],
                [0.90054, -0.11454, 0.09364],
                [0.91081, -0.11987, 0.06210],
                [0.91907, -0.07500, 0.00730],
            ],
        ),
        (
            2,
            0.0,
            -100.0,
            [
                [0.26266, 0.02014, -0.01177],
                [0.33571, 0.00960, -0.05520],
                [0.31934, 0.02834, -0.04947],
                [0.34241, 0.06962, 0.00003],
                [0.30510, 0.07079, 0.03694],
                [0.43631, 0.00715, -0.05622],
                [0.38615, 0.05174, -0.03086],
                [0.52034, 0.02930, -0.05094],
                [0.50536, 0.06699, 0.03553],
                [0.58362, 0.04318, 0.06749],
                [0.75193, 0.00382, -0.04961],
                [0.74780, 0.07018, 0.00002],
                [0.69137, 0.04446, 0.07104],
                [0.75783, -0.06118, 0.00001],
                [0.84841, -0.01710, -0.03974],
                [0.86393, 0.00313, -0.04335],
                [0.79172, 0.00161, 0.08418],
                [0.79883, -0.04136, 0.07144],
                [0.82067, -0.06899, 0.03947],
                [0.90969, -0.03556, -0.02317],
                [0.84681, 0.00146, 0.08621],
                [0.85264, -0.04252, 0.07346],
                [0.87263, -0.07162, 0.04103],
                [0.91151, -0.05652, 0.00001],
            ],
        ),
        (
            2,
            0.0,
            100.0,
            [
                [0.23726, 0.04960, -0.02813],
                [0.14419, -0.27288, -0.16702],
                [0.19947, 0.05620, -0.12950],
                [0.35768, 0.14601, -0.00006],
                [0.38814, 0.11342, 0.06964],
                [0.22414, -0.14160, -0.18736],
                [0.31471, 0.14234, -0.07935],
                [0.36260, 0.08973, -0.15402],
                [0.58683, 0.11508, 0.07033],
                [0.69880, 0.06210, 0.11407],
                [0.47773, -0.06529, -0.21749],
                [0.75028, 0.14596, -0.00002],
                [0.79144, 0.06111, 0.11083],
                [0.74004, -0.16755, -0.00010],
                [0.53781, -0.31416, -0.21552],
                [0.44835, -0.21476, -0.29486],
                [0.88927, -0.00147, 0.12660],
                [0.88583, -0.06407, 0.11087],
                [0.87199, -0.11357, 0.06586],
                [0.72709, -0.29269, -0.12035],
                [0.93393, -0.00126, 0.12452],
                [0.93137, -0.06277, 0.10860],
                [0.91893, -0.10995, 0.06368],
                [0.88156, -0.18795, -0.00012],
            ],
        ),
    ];

    /// Mean and worst OkLab error against Lightroom Classic's round-17 calibration exports, the matrix applied to its
    /// baseline patches (each export moves one primary's hue or saturation to ±100).
    fn r17_calibration_error(process: lcc_develop::Process) -> (f32, f32) {
        let (mut sum, mut worst, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for (prim, hue, sat, lrc) in R17_CAL.iter() {
            let mut cal = Calibration::default();
            match prim {
                0 => (cal.red_hue, cal.red_sat) = (*hue, *sat),
                1 => (cal.green_hue, cal.green_sat) = (*hue, *sat),
                _ => (cal.blue_hue, cal.blue_sat) = (*hue, *sat),
            }
            let m = calibration_matrix(&cal, process).unwrap();
            let mut e = 0.0f32;
            for (b, l) in R17_BASE.iter().zip(lrc.iter()) {
                let o = oklab_from_2020(calibrate(oklab_to_2020(*b), Some(&m), 0.0));
                e += ((o[0] - l[0]).powi(2) + (o[1] - l[1]).powi(2) + (o[2] - l[2]).powi(2)).sqrt();
            }
            e /= R17_BASE.len() as f32;
            sum += e;
            worst = worst.max(e);
            n += 1.0;
        }
        (sum / n, worst)
    }

    #[test]
    fn v9_calibration_follows_lightroom_classic_on_our_charts() {
        let (v8, v8_worst) = r17_calibration_error(lcc_develop::Process::V8);
        let (v9, v9_worst) = r17_calibration_error(lcc_develop::Process::V9);
        eprintln!("round 17 calibration, mean / worst OkLab error: V8 {v8:.4} / {v8_worst:.4}, V9 {v9:.4} / {v9_worst:.4}");
        // measured: V8 0.042 / 0.104, V9 0.020 / 0.058 (24 strongly coloured patches, every primary at ±100)
        assert!(v9 < 0.025 && v9_worst < 0.07, "V9 {v9} / {v9_worst}");
        assert!(v9 < 0.6 * v8 && v9_worst < 0.7 * v8_worst, "V9 {v9} / {v9_worst} vs V8 {v8} / {v8_worst}");
    }

    #[test]
    fn v9_calibration_keeps_white_and_v8_keeps_its_matrix() {
        let cal = Calibration { red_hue: 60.0, green_sat: 100.0, blue_hue: -100.0, blue_sat: -100.0, ..Default::default() };
        let m9 = calibration_matrix(&cal, lcc_develop::Process::V9).unwrap();
        for g in [0.01, 0.3, 4.0] {
            let o = calibrate([g; 3], Some(&m9), 0.0);
            assert!(o.iter().all(|v| (v - g).abs() < 1e-4 * g.max(1.0)), "{g} -> {o:?}");
        }
        assert!(calibration_matrix(&Calibration::default(), lcc_develop::Process::V9).is_none());
        // V1–V8 keep the OkLCh construction
        for p in [lcc_develop::Process::V1, lcc_develop::Process::V8] {
            assert_eq!(calibration_matrix(&cal, p), calibration_matrix(&cal, lcc_develop::Process::V7));
            assert_ne!(calibration_matrix(&cal, p), Some(m9));
        }
        // the full slider range, odd values included, stays finite
        for v in [-100.0, -99.9, 0.5, 100.0, 1e9, -1e9, f64::NAN] {
            let c = Calibration { red_hue: v, red_sat: v, green_hue: v, green_sat: v, blue_hue: v, blue_sat: v, ..Default::default() };
            if let Some(m) = calibration_matrix(&c, lcc_develop::Process::V9) {
                assert!(m.iter().flatten().all(|x| x.is_finite()), "{v}: {m:?}");
            }
        }
    }

    #[test]
    fn shadow_tint_targets_shadows() {
        let dark = calibrate([0.01, 0.01, 0.01], None, 1.0);
        let bright = calibrate([0.8, 0.8, 0.8], None, 1.0);
        assert!(dark[1] < dark[0] * 0.85, "magenta shadows: {dark:?}");
        assert!((bright[1] - bright[0]).abs() < 1e-3, "{bright:?}");
        let y = |c: [f32; 3]| lcc_color::luminance_2020(c);
        assert!((y(dark) - 0.01).abs() < 1e-5);
        let green = calibrate([0.01, 0.01, 0.01], None, -1.0);
        assert!(green[1] > green[0]);
    }

    #[test]
    fn point_color_targets_its_range() {
        let skin = [0.35, 0.16, 0.09];
        let [l, c, h] = lab_to_lch(oklab_from_2020(skin));
        let sample =
            PointColor { lum: l as f64, chroma: c as f64, hue: (h as f64).to_degrees(), hue_shift: 60.0, sat_shift: -50.0, ..Default::default() };
        let mut s = DevelopSettings::default();
        s.point_colors.push(sample);
        let ops = ColorOps::new(&s);
        let out = lab_to_lch(oklab_from_2020(ops.apply(skin, 0.0, 0.0)));
        assert!(out[1] < c * 0.7, "chroma {c} -> {}", out[1]);
        assert!(wrap(out[2] - h) > 0.2, "hue {h} -> {}", out[2]);
        // far colours are untouched
        for far in [[0.02, 0.05, 0.4], [0.05, 0.3, 0.05], [0.3, 0.3, 0.3]] {
            let o = ops.apply(far, 0.0, 0.0);
            assert!(far.iter().zip(&o).all(|(a, b)| (a - b).abs() < 1e-4), "{far:?} -> {o:?}");
        }
        // weight: 1 at the sample, falling to 0 outside the range; a wider range reaches further
        let k = PointK::new(&sample);
        assert!((k.weight(l, c, h) - 1.0).abs() < 1e-6);
        assert_eq!(k.weight(l, c, h + 1.0), 0.0);
        let wide = PointK::new(&PointColor { range: 100.0, ..sample });
        assert!(wide.weight(l, c, h + 0.45) > k.weight(l, c, h + 0.45));
    }

    #[test]
    fn point_color_variance_compresses_towards_the_sample() {
        let p = PointColor { lum: 0.6, chroma: 0.1, hue: 40.0, variance: -100.0, range: 100.0, ..Default::default() };
        let k = PointK::new(&p);
        let (h0, near) = (40f32.to_radians(), 40f32.to_radians() + 0.1);
        let o = k.apply([0.62, 0.11, near]);
        assert!((o[2] - h0).abs() < 0.1 && (o[1] - 0.1).abs() < 0.01 && (o[0] - 0.6).abs() < 0.02, "{o:?}");
        let k = PointK::new(&PointColor { variance: 100.0, ..p });
        let o = k.apply([0.62, 0.11, near]);
        assert!(o[2] - h0 > 0.1, "{o:?}");
    }

    #[test]
    fn grading_tints_shadows_only() {
        let mut s = DevelopSettings::default();
        s.grading.shadows = lcc_develop::Wheel { hue: 220.0, sat: 60.0, lum: 0.0 };
        let ops = ColorOps::new(&s);
        let dark = ops.apply([0.01, 0.01, 0.01], 0.0, 0.0);
        let bright = ops.apply([0.8, 0.8, 0.8], 0.0, 0.0);
        assert!(dark[2] > dark[0]);
        assert!((bright[2] - bright[0]).abs() < 0.02);
    }

    #[test]
    fn mask_grading_tints_its_zone_and_scales_with_the_mask() {
        let a = lcc_develop::LocalAdjustments { grade_shadows_hue: 220.0, grade_shadows_sat: 60.0, ..Default::default() };
        let full = local_grade_terms(&a);
        let dark = local_grade([0.01, 0.01, 0.01], &full);
        let bright = local_grade([0.8, 0.8, 0.8], &full);
        assert!(dark[2] > dark[0], "shadows turn blue: {dark:?}");
        assert!((bright[2] - bright[0]).abs() < 0.02, "highlights keep their colour: {bright:?}");
        // half covered by the mask: half the tint
        let half = local_grade([0.01, 0.01, 0.01], &full.map(|v| v * 0.5));
        assert!(half[2] - half[0] > 0.0 && half[2] - half[0] < dark[2] - dark[0]);
        // neutral wheels change nothing
        let none = local_grade_terms(&lcc_develop::LocalAdjustments::default());
        assert_eq!(local_grade([0.3, 0.2, 0.1], &none), [0.3, 0.2, 0.1]);
    }

    #[test]
    fn a_mask_from_before_grading_reads_as_ungraded() {
        let a: lcc_develop::LocalAdjustments =
            serde_json::from_str(r#"{"exposure": 0.5, "color_hue": 0.0, "color_sat": 0.0, "amount": 100.0}"#).unwrap();
        assert_eq!(a.exposure, 0.5);
        assert!(a.grading_wheels().iter().all(|w| w.hue == 0.0 && w.sat == 0.0 && w.lum == 0.0));
        assert!(local_grade_terms(&a).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_color_wheels_are_asc_cdl_sop() {
        // the published shape: lift pivots at white, gain at black, gamma a power, offset adds
        let full = lcc_develop::Wheel { hue: 0.0, sat: 0.0, lum: 100.0 };
        let of = |wheel: lcc_develop::Wheel, set: fn(lcc_develop::Primaries, lcc_develop::Wheel) -> lcc_develop::Primaries| {
            PrimariesK::new(&set(lcc_develop::Primaries::default(), wheel)).unwrap()
        };
        // lift's master at full: blacks move by the reach, whites stay exactly
        let k = of(full, |s, w| lcc_develop::Primaries { lift: w, ..s });
        let white = k.apply([1.0, 1.0, 1.0]);
        assert!(white.iter().all(|v| (v - 1.0).abs() < 1e-6), "lift pivots at white: {white:?}");
        let black = k.apply([0.0, 0.0, 0.0]);
        assert!(black.iter().all(|v| (v - PRIMARY_LIFT_REACH).abs() < 1e-6), "lift moves blacks by its reach: {black:?}");
        // gain: black stays exactly, everything above scales
        let k = of(full, |s, w| lcc_develop::Primaries { gain: w, ..s });
        let black = k.apply([0.0, 0.0, 0.0]);
        assert!(black.iter().all(|v| *v == 0.0), "gain pivots at black: {black:?}");
        let mid = k.apply([0.5, 0.5, 0.5]);
        assert!(mid.iter().all(|v| (v - 0.5 * (1.0 + PRIMARY_GAIN_REACH)).abs() < 1e-6), "gain multiplies: {mid:?}");
        // gamma: a power — 0 and 1 stay, the middle moves hard
        let k = of(full, |s, w| lcc_develop::Primaries { gamma: w, ..s });
        let ends = [k.apply([0.0, 0.0, 0.0]), k.apply([1.0, 1.0, 1.0])];
        assert!(ends.iter().all(|c| c.iter().all(|v| *v == 0.0 || (v - 1.0).abs() < 1e-5)), "gamma keeps black and white: {ends:?}");
        let mid = k.apply([0.25, 0.25, 0.25]);
        assert!(mid.iter().all(|v| *v > 0.35), "gamma lifts midtones hard: {mid:?}");
        // offset: everything, equally
        let k = of(full, |s, w| lcc_develop::Primaries { offset: w, ..s });
        let out = k.apply([0.4, 0.4, 0.4]);
        assert!(out.iter().all(|v| (v - 0.4 - PRIMARY_OFFSET_REACH).abs() < 1e-6), "{out:?}");
    }

    #[test]
    fn the_log_wheels_carry_their_own_strengths() {
        // the Log set's calibration: shadows halved, midtones as tuned, highlights ×3 (a multiply,
        // kept under the cliff — a full pull never clamps its zone to black)
        let full = lcc_develop::Wheel { hue: 0.0, sat: 100.0, lum: 100.0 };
        let fl = LOG_STRENGTH;
        let s = lcc_develop::Primaries {
            log_wheels: lcc_develop::LogWheels { shadow: full, midtone: full, highlight: full, offset: full },
            ..Default::default()
        };
        let k = PrimariesK::new(&s).unwrap();
        assert!((k.highlight.master - PRIMARY_GAIN_MASTER * fl[2]).abs() < 1e-6);
        assert!((k.midtone.rgb[0] / k.shadow.rgb[0] - fl[1] / fl[0]).abs() < 1e-3);
        // a full pull on every wheel stays finite on real values
        for c in [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]] {
            let out = k.apply(c);
            assert!(out.iter().all(|v| v.is_finite()), "{c:?} -> {out:?}");
        }
        // the highlight wheel alone is a graded reach, never a blow-out: a full pull on a bright
        // pixel stays above black (the multiplier keeps 1 − reach positive), a full push under ×2
        let mut s = lcc_develop::Primaries::default();
        s.log_wheels.highlight = lcc_develop::Wheel { hue: 0.0, sat: 0.0, lum: -100.0 };
        let down = PrimariesK::new(&s).unwrap();
        assert!(down.apply([0.95, 0.95, 0.95]).iter().all(|v| *v > 0.0), "no clamp to black");
        s.log_wheels.highlight = lcc_develop::Wheel { hue: 0.0, sat: 0.0, lum: 100.0 };
        let up = PrimariesK::new(&s).unwrap();
        assert!(up.apply([0.95, 0.95, 0.95]).iter().all(|v| *v < 2.0), "no leap to white");
    }

    #[test]
    fn hsl_curves_neutral_when_empty_and_targeted_when_not() {
        assert!(HslK::new(&lcc_develop::HslCurves::default()).is_none());
        // a curve on one channel only still counts
        let mut s = DevelopSettings::default();
        s.hsl_curves.sat = vec![lcc_geom::Point::new(0.5, 1.0)];
        assert!(!ColorOps::new(&s).is_identity());
        // in a black-and-white rendition the curves sit out, as the mixer does
        let bw = DevelopSettings { treatment: lcc_develop::Treatment::Bw, ..s.clone() };
        assert!(ColorOps::new(&bw).hsl.is_none());
        // the spline passes through the point itself; the LUT samples it
        let pts = [lcc_geom::Point::new(0.25, -1.0)];
        assert!((hsl_curve_at(&pts, 0.25) + 1.0).abs() < 1e-9);
        assert!((hsl_curve_at(&pts, 0.0)).abs() < 1e-9, "the hue axis's ends stay neutral");
        let k = HslK::new(&lcc_develop::HslCurves { sat: pts.to_vec(), ..Default::default() }).unwrap();
        let [_, ds, _] = k.at(0.25);
        assert!(ds < -0.9, "the LUT reads the point: {ds}");
        let [_, ds0, _] = k.at(0.02);
        assert!(ds0.abs() < 0.3, "away from the point the curve eases out: {ds0}");
    }

    #[test]
    fn an_hsl_saturation_curve_touches_only_its_hue() {
        // a bump pinned between two flat points: the hue inside gains, a hue well outside stays
        let mut s = DevelopSettings::default();
        s.hsl_curves.sat = vec![lcc_geom::Point::new(0.3, 0.0), lcc_geom::Point::new(0.55, 0.9)];
        let ops = ColorOps::new(&s);
        let of_hue = |x: f32| {
            let lab = lch_to_lab([0.5, 0.15, x * std::f32::consts::TAU]);
            oklab_to_2020(lab)
        };
        let inside = of_hue(0.55);
        let outside = of_hue(0.05);
        let chroma = |rgb: [f32; 3]| lab_to_lch(oklab_from_2020(rgb))[1];
        let i = chroma(ops.apply(inside, 0.0, 0.0));
        let o = chroma(ops.apply(outside, 0.0, 0.0));
        assert!(i > chroma(inside) * 1.5, "the hue under the point gains: {} -> {i}", chroma(inside));
        assert!((o - chroma(outside)).abs() < 1e-3, "a hue beyond the pinned flat stays: {o}");
    }

    #[test]
    fn hsl_curve_points_from_outside_clamp_and_cap() {
        let v = DevelopSettings::from_untrusted_json(&serde_json::json!({
            "hsl_curves": {"hue": [[-1.0, 5.0], [2.0, -9.0]], "lum": [[0.5, 0.25]]}
        }))
        .unwrap();
        let hue = &v.hsl_curves.hue;
        assert!(hue.iter().all(|p| (0.0..=1.0).contains(&p.x) && (-1.0..=1.0).contains(&p.y)), "{hue:?}");
        // a hostile pile of points is capped on read, as the tone curve's are
        let many: Vec<[f64; 2]> = (0..600).map(|i| [i as f64 / 600.0, 0.5]).collect();
        let v = DevelopSettings::from_untrusted_json(&serde_json::json!({"hsl_curves": {"sat": many}})).unwrap();
        assert!(v.hsl_curves.sat.len() <= lcc_develop::MAX_CURVE_POINTS);
    }

    #[test]
    fn primaries_neutral_is_identity() {
        assert!(PrimariesK::new(&lcc_develop::Primaries::default()).is_none());
        // the ranges alone move nothing, whatever they are set to
        let p = lcc_develop::Primaries { low: 0.0, high: 1.0, ..Default::default() };
        assert!(PrimariesK::new(&p).is_none());
        // a Log wheel alone is an edit like any Color one
        let mut s = DevelopSettings::default();
        s.primaries.log_wheels.shadow = lcc_develop::Wheel { hue: 220.0, sat: 40.0, lum: 0.0 };
        assert!(!ColorOps::new(&s).is_identity());
        // a hue with no saturation is no move either (as the grading wheels read it)
        let mut s = DevelopSettings::default();
        s.primaries.lift.hue = 220.0;
        assert!(ColorOps::new(&s).is_identity());
    }

    #[test]
    fn a_lift_puck_tints_shadows_harder_than_highlights() {
        let mut s = DevelopSettings::default();
        s.primaries.lift = lcc_develop::Wheel { hue: 220.0, sat: 100.0, lum: 0.0 };
        let ops = ColorOps::new(&s);
        let p = ops.primaries.as_ref().unwrap();
        let dark = p.apply([0.02, 0.02, 0.02]);
        let bright = p.apply([0.8, 0.8, 0.8]);
        let white = p.apply([1.0, 1.0, 1.0]);
        assert!(dark[2] > dark[0] + 0.1, "shadows turn blue hard: {dark:?}");
        assert!(bright[2] - bright[0] < dark[2] - dark[0], "the tint fades towards white: {bright:?}");
        assert!(white.iter().all(|v| (v - 1.0).abs() < 1e-5), "white pivots: {white:?}");
    }

    #[test]
    fn a_gain_puck_scales_proportionally() {
        let mut s = DevelopSettings::default();
        s.primaries.gain = lcc_develop::Wheel { hue: 220.0, sat: 100.0, lum: 0.0 };
        let p = PrimariesK::new(&s.primaries).unwrap();
        let black = p.apply([0.0, 0.0, 0.0]);
        assert!(black.iter().all(|v| *v == 0.0), "gain pivots at black: {black:?}");
        let low = p.apply([0.2, 0.2, 0.2]);
        let high = p.apply([0.8, 0.8, 0.8]);
        // a multiply: every level scales by its channel's factor, blue harder than red
        assert!(high[2] / 0.8 - low[2] / 0.2 < 1e-4 && high[2] / 0.8 > high[0] / 0.8, "{low:?} {high:?}");
    }

    #[test]
    fn primaries_offset_moves_everything_equally() {
        let mut s = DevelopSettings::default();
        s.primaries.offset = lcc_develop::Wheel { hue: 0.0, sat: 0.0, lum: 100.0 };
        let p = PrimariesK::new(&s.primaries).unwrap();
        for c in [[0.02, 0.02, 0.02], [0.5, 0.4, 0.3], [0.95, 0.95, 0.95]] {
            let out = p.apply(c);
            let d = out[0] - c[0];
            assert!(out.iter().zip(&c).all(|(o, i)| ((o - i) - d).abs() < 1e-6), "{c:?} -> {out:?}");
            assert!(d > 0.0);
        }
    }

    #[test]
    fn the_log_wheels_stay_narrow_and_both_sets_stack() {
        // the same pull on lift (Color, wide) and shadow (Log, narrow): the wide one reaches well
        // into the midtones, the narrow one stops below them — and both together stack additively
        let pull = lcc_develop::Wheel { hue: 220.0, sat: 100.0, lum: 0.0 };
        let s = lcc_develop::Primaries { lift: pull, ..Default::default() };
        let wide = PrimariesK::new(&s).unwrap();
        let s2 = lcc_develop::Primaries { log_wheels: lcc_develop::LogWheels { shadow: pull, ..Default::default() }, ..Default::default() };
        let log = PrimariesK::new(&s2).unwrap();
        let mid = [0.5, 0.5, 0.5];
        let w = wide.apply(mid);
        let l = log.apply(mid);
        assert!(w[2] - w[0] > (l[2] - l[0]) + 0.02, "wide {w:?} vs narrow {l:?}");
        // the narrow shadows still reach true shadows
        let dark = log.apply([0.02, 0.02, 0.02]);
        assert!(dark[2] > dark[0] + 0.02, "{dark:?}");
        // complimentary: with both sets pulled, the midtones see (at least) the Color wheels' tilt
        let both =
            lcc_develop::Primaries { lift: pull, log_wheels: lcc_develop::LogWheels { shadow: pull, ..Default::default() }, ..Default::default() };
        let k = PrimariesK::new(&both).unwrap();
        let stacked_mid = k.apply(mid);
        assert!(stacked_mid[2] - stacked_mid[0] > 0.0, "both sets render together: {stacked_mid:?}");
    }

    #[test]
    fn primaries_words_round_trip_the_gpu_layout() {
        let p = lcc_develop::Primaries {
            lift: lcc_develop::Wheel { hue: 220.0, sat: 50.0, lum: -10.0 },
            gain: lcc_develop::Wheel { hue: 0.0, sat: 25.0, lum: 15.0 },
            log_wheels: lcc_develop::LogWheels {
                shadow: lcc_develop::Wheel { hue: 220.0, sat: 50.0, lum: -10.0 },
                highlight: lcc_develop::Wheel { hue: 0.0, sat: 25.0, lum: 15.0 },
                ..Default::default()
            },
            low: 0.25,
            high: 0.8,
            ..Default::default()
        };
        let k = PrimariesK::new(&p).unwrap();
        let w = k.words();
        assert_eq!(w.len(), PRIMARY_WORDS);
        assert_eq!([w[0], w[1], w[2], w[3]], [k.lift.rgb[0], k.lift.rgb[1], k.lift.rgb[2], k.lift.master]);
        assert_eq!([w[8], w[9], w[10], w[11]], [k.gain.rgb[0], k.gain.rgb[1], k.gain.rgb[2], k.gain.master]);
        assert_eq!([w[16], w[17], w[18], w[19]], [k.shadow.rgb[0], k.shadow.rgb[1], k.shadow.rgb[2], k.shadow.master]);
        assert_eq!([w[28], w[29], w[30], w[31]], [k.log_offset.rgb[0], k.log_offset.rgb[1], k.log_offset.rgb[2], k.log_offset.master]);
        assert_eq!((w[32], w[33]), (k.low, k.high));
    }

    #[test]
    fn primaries_hostile_settings_stay_finite_and_in_range() {
        // from_untrusted_json clamps through the control specs (a sat of inf reads as the default);
        // an older experimental `log` switch in the JSON is left unread
        let v = DevelopSettings::from_untrusted_json(&serde_json::json!({
            "primaries": {
                "lift": {"hue": -900.0, "sat": 1e300, "lum": -1e300},
                "gain": {"hue": 1e308, "sat": -50.0, "lum": 1e9},
                "log": true,
                "logWheels": {"shadow": {"hue": 1e9, "sat": 1e9, "lum": -1e9}},
                "low": -2.0, "high": 99.0
            }
        }))
        .unwrap();
        let p = &v.primaries;
        assert!((0.0..=360.0).contains(&p.lift.hue) && (0.0..=100.0).contains(&p.lift.sat) && (-100.0..=100.0).contains(&p.lift.lum));
        assert!((0.0..=1.0).contains(&p.low) && (0.0..=1.0).contains(&p.high));
        assert!((-100.0..=100.0).contains(&p.log_wheels.shadow.lum), "the Log wheels clamp too");
        let k = PrimariesK::new(p).unwrap();
        for c in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.9]] {
            let out = k.apply(c);
            assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "{c:?} -> {out:?}");
        }
    }
}
