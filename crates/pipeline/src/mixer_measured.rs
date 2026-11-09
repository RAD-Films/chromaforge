//! Process V9's Color Mixer: the measured response of Lightroom Classic's Hue, Saturation and
//! Luminance sliders, as tables over hue × saturation × value in HSV of linear ProPhoto RGB (the
//! hue/saturation/value space of the public DNG spec's hue/sat maps).
//!
//! Ported from PhotoCraft (<https://github.com/storytold/photocraft>, pull request #12228, commit
//! 46ff8b8e0835b3376f8f6f303b92c626cb8904bb: `crates/algo/src/camera_raw/color_mixer.rs` and
//! `hsv_table.rs`, by fusaocom, MIT OR Apache-2.0), with its data file `data/camera_raw_hsl.bin`
//! unchanged. PhotoCraft measured the tables as a black box on Photoshop 2026's Camera Raw Filter
//! (Camera Raw 18.6), sending a 33³ RGB cube through it at every slider of every band at ±25, ±50,
//! ±75 and ±100; no Adobe code or data was used. What it found:
//!
//! - each kind acts in HSV of linear ProPhoto: the hue shift depends on the input hue, a table
//!   over hue × saturation × value gives the saturation and value changes;
//! - the bands do not add their effects: per hue, one kind's sliders combine into one effective
//!   value `u(h) = Σ w_b(h) · v_b` (band weights summing to 1 at every hue), and the change is that
//!   of every band at `u`;
//! - the kinds act one after the other, Luminance, then Saturation, then Hue, each looked up at the
//!   input's hue and at the saturation and value the previous one left.
//!
//! The same tables describe Lightroom Classic: on our own round-12 chart exports (one slider at a
//! time, 8 Nov 2026) they predict its hue shift within about 1°, its saturation within 1 % and its
//! lightness within 0.001 OkLab L, side effects included (a hue slider's change in saturation and
//! lightness), where V8's band model missed by 2–4°; on two edited sunset raws they cut the
//! mixer's error against Lightroom Classic's export by two thirds.
//!
//! CF changes: the tables apply to the display-linear Rec.2020 colour of [`ColorOps`]
//! (converted to linear ProPhoto and back, Bradford D65 ↔ D50), a mask's mixer blends its result
//! by the mask's opacity, and the GPU kernel reads the same tables ([`Measured::words`]).
//!
//! [`ColorOps`]: crate::colorops::ColorOps

use std::sync::{Arc, OnceLock};

/// Hue nodes (every 5°), saturation and value nodes over 0 … 1.
pub const HUES: usize = 72;
pub const SATS: usize = 9;
pub const VALS: usize = 9;
/// Nodes in one table.
pub const NODES: usize = HUES * SATS * VALS;
/// Tables applied one after the other: Luminance, Saturation, Hue.
pub const STEPS: usize = 3;
/// Words of a [`Measured`] mixer for the GPU kernel: per step a flag (any change) then the
/// step's nodes (hue shift in degrees, saturation change, log-value change).
pub const WORDS: usize = STEPS * (1 + NODES * 3);

/// The slider values that have tables (0 has none: no change).
const KNOTS: [f32; 8] = [-100.0, -75.0, -50.0, -25.0, 25.0, 50.0, 75.0, 100.0];
/// A colour with no more saturation than this (rounding error of a grey) is left alone.
const GREY: f32 = 1e-4;

static DATA: &[u8] = include_bytes!("../data/camera_raw_hsl.bin");

/// The measured model: per kind (hue, saturation, luminance) the band weights at every hue node
/// and a table per knot.
struct Model {
    /// `[kind][band][hue node]`.
    weights: Vec<[Vec<f32>; 8]>,
    /// `[kind][knot]`, [`NODES`] entries each.
    tables: Vec<Vec<Vec<[f32; 3]>>>,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }

    fn u32s(&mut self, n: usize) -> Option<Vec<u32>> {
        Some(self.take(n.checked_mul(4)?)?.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
    }

    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        let v: Vec<f32> = self.take(n.checked_mul(4)?)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        v.iter().all(|x| x.is_finite()).then_some(v)
    }

    /// Three scales, then per node three signed bytes times their scale.
    fn table(&mut self) -> Option<Vec<[f32; 3]>> {
        let scale = self.f32s(3)?;
        let raw = self.take(NODES.checked_mul(3)?)?;
        let s = |c: usize| scale.get(c).copied().unwrap_or(0.0);
        Some(raw.as_chunks::<3>().0.iter().map(|n| [0, 1, 2].map(|c| f32::from(n[c] as i8) * s(c))).collect())
    }
}

/// Reads `b`: `PCHS`, version 1, the node counts (hues, saturations, values) and the knot count as
/// u32; the weights as f32 `[3][8][hues]`; then per kind and knot a table.
fn parse(b: &[u8]) -> Option<Model> {
    let mut r = Reader { b, at: 0 };
    if r.take(4)? != b"PCHS" {
        return None;
    }
    if r.u32s(5)? != [1, HUES as u32, SATS as u32, VALS as u32, KNOTS.len() as u32] {
        return None;
    }
    let mut weights = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut kind: [Vec<f32>; 8] = Default::default();
        for band in &mut kind {
            *band = r.f32s(HUES)?;
        }
        weights.push(kind);
    }
    let mut tables = Vec::with_capacity(3);
    for _ in 0..3 {
        tables.push(KNOTS.iter().map(|_| r.table()).collect::<Option<Vec<_>>>()?);
    }
    (r.at == b.len()).then_some(Model { weights, tables })
}

fn model() -> Option<&'static Model> {
    static M: OnceLock<Option<Model>> = OnceLock::new();
    M.get_or_init(|| parse(DATA)).as_ref()
}

/// One kind's node `node` with every band at `u` (−100 … 100), between the measured knots.
fn at_value(tables: &[Vec<[f32; 3]>], u: f32, node: usize) -> [f32; 3] {
    if !u.is_finite() || u == 0.0 {
        return [0.0; 3];
    }
    let f = (u.clamp(-100.0, 100.0) + 100.0) / 25.0;
    let i = (f as usize).min(7);
    let t = f - i as f32;
    // knot `j` of −100, −75 … 0 … 100; 0 has no table
    let table = |j: usize| match j {
        0..4 => Some(j),
        4 => None,
        _ => Some(j - 1),
    };
    let at = |k: Option<usize>| k.and_then(|k| tables.get(k)).and_then(|t| t.get(node)).copied().unwrap_or([0.0; 3]);
    let (a, b) = (at(table(i)), at(table(i + 1)));
    [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t)
}

/// A V9 Color Mixer: its three tables (Luminance, Saturation, Hue), each [`NODES`] entries or
/// empty when that kind's sliders are all 0.
#[derive(Debug, PartialEq)]
pub struct Measured {
    steps: [Vec<[f32; 3]>; STEPS],
    /// Linear Rec.2020 (D65) → linear ProPhoto (D50) and back.
    to_pp: [[f32; 3]; 3],
    from_pp: [[f32; 3]; 3],
}

/// The mixer for slider values `hue`, `sat`, `lum` (each band −1 … 1); `None` when all are 0 (or
/// the tables could not be read, which a test rules out).
pub fn measured(hue: &[f32; 8], sat: &[f32; 8], lum: &[f32; 8]) -> Option<Arc<Measured>> {
    let off = |k: &[f32; 8]| k.iter().all(|v| *v == 0.0 || !v.is_finite());
    let sliders = [hue, sat, lum];
    if sliders.iter().all(|k| off(k)) {
        return None;
    }
    let m = model()?;
    let mut steps: [Vec<[f32; 3]>; STEPS] = Default::default();
    for (kind, ((values, weights), tables)) in sliders.iter().zip(&m.weights).zip(&m.tables).enumerate() {
        if off(values) {
            continue;
        }
        let Some(out) = steps.get_mut(STEPS - 1 - kind) else { continue };
        *out = vec![[0.0; 3]; NODES];
        for h in 0..HUES {
            let u: f32 = values
                .iter()
                .zip(weights)
                .map(|(v, w)| if v.is_finite() { (v * 100.0).clamp(-100.0, 100.0) * w.get(h).copied().unwrap_or(0.0) } else { 0.0 })
                .sum();
            for node in h * SATS * VALS..(h + 1) * SATS * VALS {
                if let Some(o) = out.get_mut(node) {
                    *o = at_value(tables, u, node);
                }
            }
        }
    }
    let to = lcc_color::REC2020.to_space(&lcc_color::PROPHOTO).0;
    let from = lcc_color::PROPHOTO.to_space(&lcc_color::REC2020).0;
    let f = |m: [[f64; 3]; 3]| m.map(|r| r.map(|v| v as f32));
    Some(Arc::new(Measured { steps, to_pp: f(to), from_pp: f(from) }))
}

fn mat(m: &[[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    m.map(|r| r[0] * c[0] + r[1] * c[1] + r[2] * c[2])
}

/// HSV (hue in degrees) of non-negative `c`.
fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let d = mx - mn;
    let h = if d <= 0.0 {
        0.0
    } else if mx == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    [h * 60.0, if mx > 0.0 { d / mx } else { 0.0 }, mx]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let f = |n: f32| {
        let k = (n + h / 60.0).rem_euclid(6.0);
        v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    [f(5.0), f(3.0), f(1.0)]
}

impl Measured {
    /// Step `k`'s change at hue `h`, saturation `s`, value `v` (trilinear; hue wraps round,
    /// saturation and value clamp to the grid).
    fn lookup(&self, k: usize, h: f32, s: f32, v: f32) -> [f32; 3] {
        let Some(t) = self.steps.get(k).filter(|t| !t.is_empty()) else { return [0.0; 3] };
        let fh = (h.rem_euclid(360.0) / 360.0 * HUES as f32).clamp(0.0, HUES as f32);
        let fs = s.clamp(0.0, 1.0) * (SATS - 1) as f32;
        let fv = v.clamp(0.0, 1.0) * (VALS - 1) as f32;
        let (ih, is, iv) = ((fh as usize).min(HUES - 1), (fs as usize).min(SATS - 2), (fv as usize).min(VALS - 2));
        let (th, ts, tv) = (fh - ih as f32, fs - is as f32, fv - iv as f32);
        let mut d = [0.0f32; 3];
        for (dh, wh) in [(0, 1.0 - th), (1, th)] {
            for (ds, ws) in [(0, 1.0 - ts), (1, ts)] {
                for (dv, wv) in [(0, 1.0 - tv), (1, tv)] {
                    let Some(n) = t.get((((ih + dh) % HUES) * SATS + is + ds) * VALS + iv + dv) else { continue };
                    let w = wh * ws * wv;
                    for c in 0..3 {
                        d[c] += w * n[c];
                    }
                }
            }
        }
        d
    }

    /// Linear ProPhoto `c` through the tables.
    fn apply_pp(&self, c: [f32; 3]) -> [f32; 3] {
        let [h0, mut s, mut v] = rgb_to_hsv(c.map(|x| x.max(0.0)));
        if v <= 1e-6 || s <= GREY {
            return c;
        }
        let (mut h, mut changed) = (h0, false);
        for k in 0..STEPS {
            let d = self.lookup(k, h0, s, v);
            if d == [0.0; 3] {
                continue;
            }
            changed = true;
            h += d[0];
            s = (s + d[1]).clamp(0.0, 1.0);
            v *= d[2].clamp(-20.0, 20.0).exp();
        }
        if !changed {
            return c;
        }
        let out = hsv_to_rgb(h, s, v);
        if out.iter().all(|x| x.is_finite()) { out } else { c }
    }

    /// Display-linear Rec.2020 `rgb` through the mixer at `strength` (0 … 1: a mask's opacity).
    pub fn apply(&self, rgb: [f32; 3], strength: f32) -> [f32; 3] {
        if strength <= 0.0 || !rgb.iter().all(|v| v.is_finite()) {
            return rgb;
        }
        let pp = mat(&self.to_pp, rgb);
        let out = self.apply_pp(pp);
        if out == pp {
            return rgb;
        }
        let back = mat(&self.from_pp, out);
        if strength >= 1.0 { back } else { [0, 1, 2].map(|c| rgb[c] + (back[c] - rgb[c]) * strength) }
    }

    /// The words the GPU kernel reads ([`WORDS`]): per step a flag (1: the step changes
    /// something), then its nodes as hue, saturation, log-value.
    pub fn words(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(WORDS);
        for t in &self.steps {
            w.push(if t.is_empty() { 0.0 } else { 1.0 });
            if t.is_empty() {
                w.extend(std::iter::repeat_n(0.0, NODES * 3));
            } else {
                w.extend(t.iter().flatten());
            }
        }
        w
    }

    /// The two colour-space matrices, row-major (Rec.2020 → ProPhoto, then back), for the kernel.
    pub fn matrices(&self) -> [f32; 18] {
        let mut m = [0.0; 18];
        for (slot, v) in m.iter_mut().zip(self.to_pp.iter().flatten().chain(self.from_pp.iter().flatten())) {
            *slot = *v;
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pp(m: &Measured, c: [f32; 3]) -> [f32; 3] {
        m.apply_pp(c)
    }

    #[test]
    fn the_tables_parse_and_bad_data_is_refused() {
        let m = model().expect("camera_raw_hsl.bin parses");
        for kind in &m.weights {
            for h in 0..HUES {
                let s: f32 = kind.iter().map(|b| b[h]).sum();
                assert!((s - 1.0).abs() < 1e-3, "weights sum to 1 at hue node {h}: {s}");
            }
        }
        assert!(parse(&DATA[..DATA.len() - 1]).is_none(), "truncated data is refused");
        assert!(parse(b"PCHS").is_none());
        assert!(parse(b"").is_none());
    }

    #[test]
    fn neutral_sliders_make_no_mixer_and_greys_stay() {
        assert!(measured(&[0.0; 8], &[0.0; 8], &[0.0; 8]).is_none());
        assert!(measured(&[f32::NAN; 8], &[0.0; 8], &[0.0; 8]).is_none());
        let m = measured(&[1.0; 8], &[-1.0; 8], &[1.0; 8]).expect("a change");
        for g in [0.0, 0.2, 0.7, 1.0] {
            assert_eq!(m.apply([g; 3], 1.0), [g; 3]);
        }
    }

    #[test]
    fn kinds_apply_luminance_then_saturation_then_hue() {
        let (mut s, mut l) = ([0.0; 8], [0.0; 8]);
        s[5] = 1.0;
        l[5] = 1.0;
        let m = measured(&[0.0; 8], &s, &l).expect("a change");
        let c = [0.05, 0.1, 0.5];
        let [h0, s0, v0] = rgb_to_hsv(c);
        let dl = m.lookup(0, h0, s0, v0);
        let (s1, v1) = ((s0 + dl[1]).clamp(0.0, 1.0), v0 * dl[2].exp());
        let ds = m.lookup(1, h0, s1, v1);
        let want = hsv_to_rgb(h0 + dl[0] + ds[0], (s1 + ds[1]).clamp(0.0, 1.0), v1 * ds[2].exp());
        let got = pp(&m, c);
        assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{got:?} vs {want:?}");
        assert_eq!(m.lookup(2, h0, s0, v0), [0.0; 3], "no hue sliders, no Hue table");
    }

    #[test]
    fn a_mask_blends_by_its_opacity_and_odd_inputs_are_safe() {
        let mut h = [0.0; 8];
        h[3] = -1.0;
        let m = measured(&h, &[0.0; 8], &[0.0; 8]).expect("a change");
        let c = [0.1, 0.4, 0.1];
        let full = m.apply(c, 1.0);
        let half = m.apply(c, 0.5);
        assert_ne!(full, c);
        assert_eq!(m.apply(c, 0.0), c);
        for i in 0..3 {
            assert!((half[i] - (c[i] + full[i]) / 2.0).abs() < 1e-6);
        }
        let big = measured(&[1e9; 8], &[-1e9; 8], &[f32::INFINITY; 8]).expect("clamped");
        for c in [[f32::NAN, 0.2, 0.1], [-1.0, 2.0, 0.5], [1e6, 0.0, 0.0], [f32::INFINITY, 0.0, 0.0], [0.0; 3]] {
            let o = big.apply(c, 1.0);
            assert!(o.iter().all(|v| v.is_finite()) || c.iter().any(|v| !v.is_finite()), "{c:?} → {o:?}");
        }
    }

    #[test]
    fn the_gpu_words_hold_every_step() {
        let mut s = [0.0; 8];
        s[2] = -0.5;
        let m = measured(&[0.0; 8], &s, &[0.0; 8]).expect("a change");
        let w = m.words();
        assert_eq!(w.len(), WORDS);
        let step = 1 + NODES * 3;
        assert_eq!((w[0], w[step], w[2 * step]), (0.0, 1.0, 0.0), "only Saturation changes");
    }
}
