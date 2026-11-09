//! The small, outline-following part of a raw's Highlights − in process V9: bright tones next to clipped areas go
//! down a little more than the picture's curve ([`crate::tone_lift`]) takes them.
//!
//! Measured on Lightroom Classic exports of our own Canon EOS R5 C raws (rounds 9b and 10, OneSlips, Nov 2026): with
//! the per-picture curve alone, pixels at the edge of a clipped sky came out 2.5 L* lighter than Lightroom's −100,
//! a light rim around the clouds, while pixels 60 px away matched. Lightroom pulls bright tones near clipped areas
//! further, following the areas' outlines. Here the clipped areas (brightest channel past the camera curve's white)
//! are spread edge-aware, guided by log luminance, over a few hundredths of the frame, and bright tones within
//! [`GATE`] EV of white get up to [`AMOUNT`] EV more pull where the spread is. The rim then sits 0.3 L* from
//! Lightroom's.
//!
//! The spread is taken once per photo on a fixed-size render of the whole frame ([`EDGE`] px long), so previews,
//! zoomed windows and exports agree, and read back bilinearly at each output pixel.

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

/// The extra pull of one photo: a plane over the whole output frame, at Highlights −100.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgePull {
    /// The pull (EV, ≤ 0) at Highlights −100, on the whole frame at [`EDGE`] px long.
    pub plane: Arc<Plane>,
    /// The Highlights slider's share of it.
    pub gain: f32,
    /// Plane pixels per output pixel, `(x, y)`.
    pub scale: (f32, f32),
    /// Where the rendered pixels sit in the whole output frame.
    pub origin: (usize, usize),
}

impl EdgePull {
    /// The pull (EV) at rendered pixel `(x, y)`, read bilinearly from the plane.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        let (pw, ph) = (self.plane.width, self.plane.height);
        if pw == 0 || ph == 0 {
            return 0.0;
        }
        let fx = ((self.origin.0.saturating_add(x) as f32 + 0.5) * self.scale.0 - 0.5).clamp(0.0, (pw - 1) as f32);
        let fy = ((self.origin.1.saturating_add(y) as f32 + 0.5) * self.scale.1 - 0.5).clamp(0.0, (ph - 1) as f32);
        let (fx, fy) = (if fx.is_finite() { fx } else { 0.0 }, if fy.is_finite() { fy } else { 0.0 });
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |x: usize, y: usize| self.plane.data.get(y * pw + x).copied().unwrap_or(0.0);
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        let v = self.gain * (top + (bottom - top) * ty);
        if v.is_finite() { v } else { 0.0 }
    }
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    if t.is_finite() { t * t * (3.0 - 2.0 * t) } else { 0.0 }
}

/// The scene tone (EV over grey) where `tm`'s displayed tone reaches its top, within 0.05 EV.
fn white_of(tm: &ToneMap) -> Option<f32> {
    let d = |e: f32| (tm.apply(GREY * e.exp2()).max(1e-9) / GREY).log2();
    let top = d(12.0);
    (0..=640).map(|i| -8.0 + i as f32 / 32.0).find(|&e| d(e) >= top - 0.05)
}

/// The pull at Highlights −100 for a white-balanced, exposed `img` (scene linear, exposure `ev` still to add) whose
/// camera curve reaches white at scene tone `white`, with `ppl` pixels per long edge.
fn unit_plane(img: &Rgb32f, ev: f32, white: f32, ppl: f64) -> Plane {
    let log_l = img.map(log_lum);
    let mut clipped = Plane::new(img.width, img.height);
    for (o, c) in clipped.data.iter_mut().zip(&img.data) {
        let m = c[0].max(c[1]).max(c[2]);
        *o = if log_lum([m, m, m]) + ev >= white { 1.0 } else { 0.0 };
    }
    let near = crate::masks::guided_cross(&log_l, &clipped, (RADIUS * ppl as f32).max(1.0), LEAK);
    let mut plane = Plane::new(img.width, img.height);
    for ((o, n), l) in plane.data.iter_mut().zip(&near.data).zip(&log_l.data) {
        let w = (n / FULL_AT).clamp(0.0, 1.0);
        let v = -AMOUNT * w * smooth(white - GATE - 1.0, white - GATE, l + ev);
        *o = if v.is_finite() { v } else { 0.0 };
    }
    plane
}

type Cached = (u64, Arc<Plane>);
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
    let white = white_of(tm)?;
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
            white.to_bits(),
        ))
    };
    let cached = LAST.lock().unwrap_or_else(PoisonError::into_inner).as_ref().filter(|(k, _)| *k == key).map(|(_, p)| p.clone());
    let plane = match cached {
        Some(p) => p,
        None => {
            let mut img = Rgb32f::clone(&frame_proxy(src, frame, pw, ph));
            white_balance(&mut img, info, s);
            let p = Arc::new(unit_plane(&img, ev, white, frame.px_per_long(pw)));
            *LAST.lock().unwrap_or_else(PoisonError::into_inner) = Some((key, p.clone()));
            p
        }
    };
    Some(Arc::new(EdgePull { plane, gain, scale: (pw as f32 / full.0 as f32, ph as f32 / full.1 as f32), origin }))
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

    #[test]
    fn reads_back_bilinearly_and_never_off_the_plane() {
        let mut plane = Plane::new(2, 1);
        plane.data = vec![0.0, -1.0];
        let e = EdgePull { plane: Arc::new(plane), gain: 0.5, scale: (0.5, 0.5), origin: (0, 0) };
        assert_eq!(e.at(0, 0), 0.0);
        assert!((e.at(2, 0) + 0.375).abs() < 1e-6, "{}", e.at(2, 0));
        assert_eq!(e.at(usize::MAX, usize::MAX), -0.5);
        let empty = EdgePull { plane: Arc::new(Plane::new(0, 0)), gain: 1.0, scale: (1.0, 1.0), origin: (0, 0) };
        assert_eq!(empty.at(3, 3), 0.0);
    }
}
