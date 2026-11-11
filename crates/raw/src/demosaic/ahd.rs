//! Adaptive Homogeneity-Directed demosaicing (K. Hirakawa, T. W. Parks, "Adaptive homogeneity-directed
//! demosaicing algorithm", IEEE Trans. Image Processing 14(3), 2005), implemented from the paper:
//!
//! 1. Interpolate green horizontally and vertically (5-tap filter [−¼ ½ ½ ½ −¼] across the missing site,
//!    clamped to the two adjacent greens), giving two candidate green planes.
//! 2. Complete each candidate with red/blue by colour-difference (C − G) interpolation.
//! 3. Convert both candidates to CIELab and measure, per pixel, how many of its 4 neighbours lie within the
//!    adaptive luminance / chrominance tolerances (ε_L, ε_C from the paper) — the homogeneity.
//! 4. Sum homogeneity over a 3×3 window and pick the more homogeneous candidate (average on ties), as the paper does
//!    ([`super::Method::Ahd`], processes V1-V8). [`super::Method::AhdBlend`] (process V9) blends the two by how far
//!    apart their sums are (see [`BLEND`]), so a clear edge still takes one candidate.
//!
//! [`super::Method::AhdBlend`] also knows clipped samples: within [`REACH`] of one, every interpolated channel stays
//! within the range of its colour's samples around the pixel ([`CLIP`]). A clipped sample reads lower than the
//! light was, so a colour difference taken against it (red minus a green interpolated from clipped greens) is far
//! too negative or too positive, and the next pixel's colour went past anything its samples allow: black specks on
//! a lamp's bright rim (a channel below zero) and neon green-cyan pixels where branches cross a white sky.
//!
//! Work is done in overlapping tiles (bounded memory, rayon-parallel), one band of tiles at a time written
//! straight into the output. Each step runs only where the tile's output depends on it: the choice reads the
//! homogeneity 1 pixel around the tile, which reads Lab 2 pixels around, the candidates read the greens 3
//! pixels around, and those read the mosaic 5 pixels around, inside the `MARGIN`.

use super::Mosaic;
use crate::Rgb32f;
use rayon::prelude::*;

const TILE: usize = 128;
const MARGIN: usize = 6;
/// Difference of the two candidates' 3×3 homogeneity sums (each 0–36) from which the more homogeneous one is taken
/// alone; closer sums blend the two linearly, equal sums half and half. In a flat, noisy area noise alone decides
/// which sum is larger, and picking outright turned the noise into short horizontal and vertical strokes (a maze,
/// coloured where white balance is strong). 12 keeps the PSNR of the outright pick on the edge tests.
const BLEND: f32 = 12.0;
/// A sample at or above this share of the file's white level counts as clipped. The mosaic is normalised to its white
/// level per colour site ([`crate::RawImage::normalized`]: 1.0 for every channel), so one level serves every colour.
/// The same share as the highlight rebuild's clip.
const CLIP: f32 = 0.99;
/// How far (px) a clipped sample reaches into the two candidates: the directional greens read the mosaic 2 px away,
/// and the colour differences read the greens 1 px away.
const REACH: usize = 3;

/// `blend`: blend the two candidates where their homogeneity sums are close ([`BLEND`], process V9's
/// [`super::Method::AhdBlend`]); false: the more homogeneous one outright, the average on ties (V1-V8).
pub(crate) fn ahd(m: &Mosaic, blend: bool) -> Rgb32f {
    ahd_with(m, blend, blend)
}

/// [`ahd`], with the candidates kept within their samples' range near clipped ones (`clip_aware`) or not.
fn ahd_with(m: &Mosaic, blend: bool, clip_aware: bool) -> Rgb32f {
    let (w, h) = (m.w, m.h);
    let mut out = Rgb32f::new(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    // one row of tiles per task, its tiles in parallel too, written straight into the picture
    out.data.par_chunks_mut((w * TILE).max(1)).enumerate().for_each(|(ty, band)| {
        let (y0, th) = (ty * TILE, band.len() / w.max(1));
        let tiles: Vec<(usize, usize, Vec<[f32; 3]>)> = (0..w.div_ceil(TILE))
            .into_par_iter()
            .map(|tx| {
                let (x0, tw) = (tx * TILE, TILE.min(w - tx * TILE));
                (x0, tw, tile(m, x0, y0, tw, th, blend, clip_aware))
            })
            .collect();
        for (x0, tw, px) in tiles {
            for (row, src) in band.chunks_exact_mut(w).zip(px.chunks_exact(tw)) {
                row[x0..x0 + tw].copy_from_slice(src);
            }
        }
    });
    out
}

/// Linear camera RGB (treated as sRGB/D65 primaries for the homogeneity metric only) → CIELab.
#[inline]
fn lab(p: [f32; 3]) -> [f32; 3] {
    let x = (0.4124 * p[0] + 0.3576 * p[1] + 0.1805 * p[2]) / 0.95047;
    let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let z = (0.0193 * p[0] + 0.1192 * p[1] + 0.9505 * p[2]) / 1.08883;
    // all three without a branch, so they run side by side (a value past 1e30 has no meaning here)
    let [fx, fy, fz] = [x, y, z].map(|t| {
        let root = cube_root(t.clamp(0.008856, 1.0e30));
        if t > 0.008856 { root } else { 7.787 * t + 16.0 / 116.0 }
    });
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// The cube root of a positive, finite `t`, within three units in the last place: a third of its exponent read
/// straight from its bits (`bits(1.0)` keeps 1 at 1), then two of Halley's steps, each of which cubes the relative
/// error. The library `cbrt` was the most expensive call of the whole demosaic.
#[inline]
fn cube_root(t: f32) -> f32 {
    const ONE_THIRD_BIAS: u32 = 0x3f80_0000 / 3 * 2;
    let mut y = f32::from_bits(t.to_bits() / 3 + ONE_THIRD_BIAS);
    for _ in 0..2 {
        let y3 = y * y * y;
        y *= (y3 + 2.0 * t) / (2.0 * y3 + t);
    }
    y
}

fn tile(m: &Mosaic, x0: usize, y0: usize, tw: usize, th: usize, blend: bool, clip_aware: bool) -> Vec<[f32; 3]> {
    let (bw, bh) = (tw + 2 * MARGIN, th + 2 * MARGIN);
    let n = bw * bh;
    let (ox, oy) = (x0 as isize - MARGIN as isize, y0 as isize - MARGIN as isize);
    let mut mos = vec![0f32; n];
    let mut col = vec![0u8; n];
    for (y, (mrow, crow)) in mos.chunks_exact_mut(bw).zip(col.chunks_exact_mut(bw)).enumerate() {
        let ry = super::reflect(oy + y as isize, m.h);
        if ox >= 0 && ox as usize + bw <= m.w {
            // inside the image: the row as it is, and its colours repeating with the pattern
            let x0 = ry * m.w + ox as usize;
            mrow.copy_from_slice(&m.data[x0..x0 + bw]);
            let period: Vec<u8> = (0..m.cfa.width).map(|k| m.cfa.color_at(ox as usize + k, ry)).collect();
            for (c, &p) in crow.iter_mut().zip(period.iter().cycle()) {
                *c = p;
            }
        } else {
            for x in 0..bw {
                mrow[x] = m.get(ox + x as isize, oy + y as isize);
                crow[x] = m.color(ox + x as isize, oy + y as isize);
            }
        }
    }
    // flat indices of the pixels `d` or more from the buffer's edge, row by row
    let inner = |d: usize| (d..bh - d).flat_map(move |y| y * bw + d..y * bw + bw - d);
    // process V9: the pixels whose candidates read a clipped sample (none in a tile without one)
    let near = if clip_aware { near_clipped(&mos, bw, bh) } else { None };
    // 1. directional greens
    let mut gh = vec![0f32; n];
    let mut gv = vec![0f32; n];
    for i in inner(MARGIN - 3) {
        let c = mos[i];
        if col[i] == 1 {
            gh[i] = c;
            gv[i] = c;
            continue;
        }
        let (l, r) = (mos[i - 1], mos[i + 1]);
        let est = (l + r) * 0.5 + (2.0 * c - mos[i - 2] - mos[i + 2]) * 0.25;
        gh[i] = super::clamp_between(est, l, r);
        let (u, d) = (mos[i - bw], mos[i + bw]);
        let est = (u + d) * 0.5 + (2.0 * c - mos[i - 2 * bw] - mos[i + 2 * bw]) * 0.25;
        gv[i] = super::clamp_between(est, u, d);
    }
    // 2. full candidates, and 3. their Lab
    // (`p` with channel `c` set to `v`, in registers: writing channels through an index and reading the pixel
    // back for its Lab stalled on store forwarding)
    let set = |p: [f32; 3], c: u8, v: f32| [if c == 0 { v } else { p[0] }, if c == 1 { v } else { p[1] }, if c >= 2 { v } else { p[2] }];
    let complete = |g: &[f32]| -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut rgb = vec![[0f32; 3]; n];
        let mut lab_of = vec![[0f32; 3]; n];
        for i in inner(MARGIN - 2) {
            let own = col[i];
            let gvv = g[i];
            let d = |j: usize| mos[j] - g[j];
            let p = if own == 1 {
                let p = set([0.0, gvv, 0.0], col[i + 1], gvv + (d(i - 1) + d(i + 1)) * 0.5);
                set(p, col[i + bw], gvv + (d(i - bw) + d(i + bw)) * 0.5)
            } else {
                let p = set([0.0, gvv, 0.0], own, mos[i]);
                set(p, 2 - own, gvv + (d(i - bw - 1) + d(i - bw + 1) + d(i + bw - 1) + d(i + bw + 1)) * 0.25)
            };
            let p = match &near {
                Some(near) if near[i] => within_samples(p, own, &mos, &col, i, bw),
                _ => p,
            };
            rgb[i] = p;
            lab_of[i] = lab(p);
        }
        (rgb, lab_of)
    };
    let (rh, lh) = complete(&gh);
    let (rv, lv) = complete(&gv);
    // homogeneity
    let dl = |a: &[f32; 3], b: &[f32; 3]| (a[0] - b[0]).abs();
    let dc = |a: &[f32; 3], b: &[f32; 3]| (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let mut hh = vec![0u8; n];
    let mut hv = vec![0u8; n];
    for i in inner(MARGIN - 1) {
        // left, right, up, down
        let (l, r, u, d) = (i - 1, i + 1, i - bw, i + bw);
        let (a, b) = (lh[i], lv[i]);
        let dlh = [dl(&a, &lh[l]), dl(&a, &lh[r]), dl(&a, &lh[u]), dl(&a, &lh[d])];
        let dch = [dc(&a, &lh[l]), dc(&a, &lh[r]), dc(&a, &lh[u]), dc(&a, &lh[d])];
        let dlv = [dl(&b, &lv[l]), dl(&b, &lv[r]), dl(&b, &lv[u]), dl(&b, &lv[d])];
        let dcv = [dc(&b, &lv[l]), dc(&b, &lv[r]), dc(&b, &lv[u]), dc(&b, &lv[d])];
        let eps_l = dlh[0].max(dlh[1]).min(dlv[2].max(dlv[3]));
        let eps_c = dch[0].max(dch[1]).min(dcv[2].max(dcv[3]));
        hh[i] = (0..4).map(|k| u8::from(dlh[k] <= eps_l && dch[k] <= eps_c)).sum();
        hv[i] = (0..4).map(|k| u8::from(dlv[k] <= eps_l && dcv[k] <= eps_c)).sum();
    }
    // 4. choose
    let mut out = Vec::with_capacity(tw * th);
    for i in inner(MARGIN) {
        let (mut sh, mut sv) = (0u32, 0u32);
        for j in [i - bw - 1, i - bw, i - bw + 1, i - 1, i, i + 1, i + bw - 1, i + bw, i + bw + 1] {
            sh += hh[j] as u32;
            sv += hv[j] as u32;
        }
        if !blend {
            out.push(if sh > sv {
                rh[i]
            } else if sv > sh {
                rv[i]
            } else {
                [(rh[i][0] + rv[i][0]) * 0.5, (rh[i][1] + rv[i][1]) * 0.5, (rh[i][2] + rv[i][2]) * 0.5]
            });
            continue;
        }
        let wh = (0.5 + (sh as f32 - sv as f32) / (2.0 * BLEND)).clamp(0.0, 1.0);
        // a candidate taken alone is copied, and one that isn't finite (beside a float raw's NaN sample) is left out,
        // so the other's value doesn't become 0 · NaN
        let finite = |p: &[f32; 3]| p.iter().all(|v| v.is_finite());
        out.push(if wh >= 1.0 || !finite(&rv[i]) {
            rh[i]
        } else if wh <= 0.0 || !finite(&rh[i]) {
            rv[i]
        } else {
            [0, 1, 2].map(|c| rh[i][c] * wh + rv[i][c] * (1.0 - wh))
        });
    }
    out
}

/// Which pixels of a `bw × bh` tile buffer have a clipped sample ([`CLIP`]) within [`REACH`] (a square), or `None`
/// when the tile has no clipped sample.
fn near_clipped(mos: &[f32], bw: usize, bh: usize) -> Option<Vec<bool>> {
    if !mos.iter().any(|&v| v >= CLIP) {
        return None;
    }
    let clipped: Vec<bool> = mos.iter().map(|&v| v >= CLIP).collect();
    let mut rows = vec![false; mos.len()];
    for (src, dst) in clipped.chunks_exact(bw).zip(rows.chunks_exact_mut(bw)) {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src.get(x.saturating_sub(REACH)..(x + REACH + 1).min(bw)).is_some_and(|s| s.contains(&true));
        }
    }
    let mut near = vec![false; mos.len()];
    for y in 0..bh {
        for x in 0..bw {
            let hit = (y.saturating_sub(REACH)..(y + REACH + 1).min(bh)).any(|yy| rows.get(yy * bw + x).copied().unwrap_or(false));
            if let Some(n) = near.get_mut(y * bw + x) {
                *n = hit;
            }
        }
    }
    Some(near)
}

/// `p` (the candidate at flat index `i` of the tile buffer, whose own colour is `own`) with each interpolated channel
/// kept within the range of that colour's samples among the 8 neighbours.
fn within_samples(mut p: [f32; 3], own: u8, mos: &[f32], col: &[u8], i: usize, bw: usize) -> [f32; 3] {
    let around = [i - bw - 1, i - bw, i - bw + 1, i - 1, i + 1, i + bw - 1, i + bw, i + bw + 1];
    for c in 0..3u8 {
        if c == own {
            continue;
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &j in &around {
            if col.get(j) == Some(&c)
                && let Some(&v) = mos.get(j)
            {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if let Some(v) = p.get_mut(c as usize) {
            *v = super::clamp_between(*v, lo, hi);
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::cube_root;

    /// Within three units in the last place of the true cube root, over the whole range the Lab conversion uses
    /// it for (every 997th float from its threshold up to 1e30).
    #[test]
    fn cube_root_is_float_precise() {
        let (lo, hi) = (0.008856f32.to_bits(), 1.0e30f32.to_bits());
        for bits in (lo..hi).step_by(997) {
            let t = f32::from_bits(bits);
            let want = f64::from(t).cbrt();
            let got = f64::from(cube_root(t));
            let near = want as f32;
            let ulp = f64::from(f32::from_bits(near.to_bits() + 1)) - f64::from(near);
            assert!((got - want).abs() <= 3.0 * ulp, "{t}: {got} vs {want}");
        }
        assert_eq!(cube_root(1.0), 1.0);
        assert_eq!(cube_root(8.0), 2.0);
    }

    use super::super::{Mosaic, mosaic_from_rgb};
    use super::{CLIP, ahd_with};
    use crate::{Cfa, Rgb32f};

    /// Branches and a lamp's rim against a light 4 stops over white: a dark textured scene with a bright disc and
    /// thin dark bars crossing it, its colour varying, at `peak` × white for the bright parts (clipped to 1.0 in the
    /// mosaic, as a sensor does).
    fn branches(peak: f32) -> (Rgb32f, Cfa) {
        let (w, h) = (96usize, 80usize);
        let img = Rgb32f::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32, y as f32);
            let disc = (fx - 48.0).powi(2) + (fy - 40.0).powi(2) < 28.0f32.powi(2);
            let bar = (x + y / 2) % 13 < 2 || (x * 3 + y) % 17 == 0;
            if disc && !bar {
                [peak * 0.9, peak, peak * 0.7]
            } else {
                let t = 0.5 + 0.5 * ((fx * 0.37).sin() * (fy * 0.23).cos());
                [0.08 + 0.3 * t, 0.12 + 0.1 * t, 0.03 + 0.05 * (1.0 - t)]
            }
        });
        let img = Rgb32f { data: img.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..img };
        (img, Cfa::bayer("RGGB").unwrap())
    }

    /// The largest amount by which a pixel's interpolated channel lies outside its colour's 8 neighbouring samples,
    /// over pixels within reach of a clipped sample (and their count).
    fn overshoot(out: &Rgb32f, mos: &[f32], cfa: &Cfa, w: usize, h: usize) -> (f32, usize) {
        let (mut worst, mut n) = (0f32, 0usize);
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let near = (y - 2..=y + 2).any(|yy| (x - 2..=x + 2).any(|xx| mos[yy * w + xx] >= CLIP));
                if !near {
                    continue;
                }
                n += 1;
                for c in 0..3u8 {
                    if cfa.color_at(x, y) == c {
                        continue;
                    }
                    let s: Vec<f32> = (y - 1..=y + 1)
                        .flat_map(|yy| (x - 1..=x + 1).map(move |xx| (xx, yy)))
                        .filter(|&(xx, yy)| (xx, yy) != (x, y) && cfa.color_at(xx, yy) == c)
                        .map(|(xx, yy)| mos[yy * w + xx])
                        .collect();
                    let (lo, hi) = s.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                    let v = out.get(x, y)[c as usize];
                    worst = worst.max(lo - v).max(v - hi);
                }
            }
        }
        (worst, n)
    }

    /// Process V9: next to clipped samples no interpolated channel goes past its colour's samples around it (plain AHD
    /// does here: a black speck or a neon pixel), and every pixel out of their reach is exactly plain AHD's.
    #[test]
    fn near_clipped_samples_the_colours_stay_within_their_samples() {
        let (img, cfa) = branches(16.0);
        let n = mosaic_from_rgb(&img, &cfa);
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        let (plain, aware) = (ahd_with(&m, true, false), ahd_with(&m, true, true));
        let (before, count) = overshoot(&plain, &n.data, &cfa, n.width, n.height);
        let (after, _) = overshoot(&aware, &n.data, &cfa, n.width, n.height);
        assert!(count > 500, "{count} pixels near clip");
        assert!(before > 0.05, "plain AHD overshoots its samples here: {before}");
        assert!(after <= 1e-6, "clip-aware: {after}");
        assert!(aware.data.iter().all(|p| p.iter().all(|v| v.is_finite() && *v >= 0.0)));
        // out of reach (the candidates' 3 px and the homogeneity's 3 more): plain AHD, bit for bit
        let (w, h) = (n.width, n.height);
        let mut far = 0;
        for y in 0..h {
            for x in 0..w {
                let reach =
                    (y.saturating_sub(6)..(y + 7).min(h)).any(|yy| (x.saturating_sub(6)..(x + 7).min(w)).any(|xx| n.data[yy * w + xx] >= CLIP));
                if !reach {
                    far += 1;
                    assert_eq!(aware.get(x, y), plain.get(x, y), "({x}, {y})");
                }
            }
        }
        assert!(far > 500, "{far} pixels out of reach");
    }

    /// A frame without a clipped sample develops exactly as plain AHD.
    #[test]
    fn an_unclipped_frame_is_unchanged() {
        let (img, cfa) = branches(0.95);
        let n = mosaic_from_rgb(&img, &cfa);
        assert!(n.data.iter().all(|&v| v < CLIP));
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        assert_eq!(ahd_with(&m, true, true).data, ahd_with(&m, true, false).data);
    }
}
