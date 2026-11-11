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
//! a lamp's bright rim (a channel below zero) and neon green-cyan pixels where branches cross a white sky. Within
//! [`RATIO_REACH`] of one, red and blue also follow green's shape across an edge: green times their ratio to green at
//! the samples they are interpolated from, not green plus their difference to it ([`by_ratio`]). From dark branches to
//! a sky near its clip that difference grows with the brightness, so a pixel between them took too little red or blue
//! from the bright side's difference: the rest of the neon rim.
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
/// How far (px) from a clipped sample red and blue follow green's shape ([`by_ratio`]): an edge in front of a clipped
/// sky is often a few pixels short of the clipping (the sky dims toward it), and the neon rim with it. The same 6 px
/// the fit-size binning reaches (`binned.rs`); with the homogeneity's 3 px, pixels up to 9 px away can change.
const RATIO_REACH: usize = 6;
/// [`by_ratio`] blends from the colour difference (green at the sites below `.0` of the white) to the ratio (above
/// `.1`): a ratio to a green that noise dominates comes out too high on average (red about 2 % around glints on a dark
/// noisy surface, a magenta cast).
const GUIDE_DARK: (f32, f32) = (0.03, 0.12);
/// [`by_ratio`] fades back to the colour difference as the pixel's own green nears the clip (from `.0` to `.1` of the
/// white): the greens around it are clipped or about to clip there, and a ratio to an understated green overstates the
/// colour. These pixels are also what the highlight rebuild takes as a clipped surface's colour just below its clip:
/// with the ratio there, a frame's partly clipped foliage in a window came out 0.4 dE further from Lightroom Classic.
/// The neon rim lies lower, between a dark edge and the sky (green 0.17 to 0.57 of the white on an R5 C frame).
const GUIDE_BRIGHT: (f32, f32) = (0.7, 0.95);
/// Added to both sides of [`by_ratio`]'s ratio (of the white), so that it stays finite at black.
const RATIO_FLOOR: f32 = 1e-3;

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
    // and those within [`RATIO_REACH`] of one, whose red and blue follow green ([`by_ratio`])
    let near_ratio = if clip_aware { near_clipped_wide(m, &mos, (ox, oy), bw, bh) } else { None };
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
            let p = match &near_ratio {
                Some(nr) if nr[i] => by_ratio(p, own, gvv, &mos, &col, g, i, bw),
                _ => p,
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
    let reach = REACH;
    if !mos.iter().any(|&v| v >= CLIP) {
        return None;
    }
    let clipped: Vec<bool> = mos.iter().map(|&v| v >= CLIP).collect();
    let mut rows = vec![false; mos.len()];
    for (src, dst) in clipped.chunks_exact(bw).zip(rows.chunks_exact_mut(bw)) {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src.get(x.saturating_sub(reach)..(x + reach + 1).min(bw)).is_some_and(|s| s.contains(&true));
        }
    }
    let mut near = vec![false; mos.len()];
    for y in 0..bh {
        for x in 0..bw {
            let hit = (y.saturating_sub(reach)..(y + reach + 1).min(bh)).any(|yy| rows.get(yy * bw + x).copied().unwrap_or(false));
            if let Some(n) = near.get_mut(y * bw + x) {
                *n = hit;
            }
        }
    }
    Some(near)
}

/// Which pixels of the `bw × bh` tile buffer `mos` (whose top-left lies at `at` of the mosaic `m`) have a clipped sample
/// ([`CLIP`]) within [`RATIO_REACH`] (a square), or `None` when none has. The candidates are completed down to `MARGIN -
/// 2` px from the buffer's edge, and their square reaches past it: the mosaic is read that much beyond the buffer.
fn near_clipped_wide(m: &Mosaic, mos: &[f32], at: (isize, isize), bw: usize, bh: usize) -> Option<Vec<bool>> {
    let extra = RATIO_REACH.saturating_sub(MARGIN - 2);
    let (ew, eh) = (bw + 2 * extra, bh + 2 * extra);
    let (ex, ey) = (at.0 - extra as isize, at.1 - extra as isize);
    let clipped_at = |x: usize, y: usize| m.get(ex + x as isize, ey + y as isize) >= CLIP;
    // (the ring outside the buffer first: a tile with no clipped sample in or around it is the rule)
    let ring = (0..eh).any(|y| {
        if y < extra || y >= eh - extra { (0..ew).any(|x| clipped_at(x, y)) } else { (0..extra).chain(ew - extra..ew).any(|x| clipped_at(x, y)) }
    });
    if !ring && !mos.iter().any(|&v| v >= CLIP) {
        return None;
    }
    let clipped: Vec<bool> = (0..eh).flat_map(|y| (0..ew).map(move |x| (x, y))).map(|(x, y)| clipped_at(x, y)).collect();
    let mut rows = vec![false; ew * eh];
    for (src, dst) in clipped.chunks_exact(ew).zip(rows.chunks_exact_mut(ew)) {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src.get(x.saturating_sub(RATIO_REACH)..(x + RATIO_REACH + 1).min(ew)).is_some_and(|s| s.contains(&true));
        }
    }
    let mut near = vec![false; bw * bh];
    for (y, row) in near.chunks_exact_mut(bw).enumerate() {
        let yy = y + extra;
        for (x, n) in row.iter_mut().enumerate() {
            let xx = x + extra;
            *n = (yy.saturating_sub(RATIO_REACH)..(yy + RATIO_REACH + 1).min(eh)).any(|r| rows.get(r * ew + xx).copied().unwrap_or(false));
        }
    }
    Some(near)
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `p` (the candidate at flat index `i` of the tile buffer, own colour `own`, green `green` from the candidate's green
/// plane `g`) with each interpolated red and blue taken as green times its ratio to green at the sites it is
/// interpolated from, instead of green plus its difference to it. Across an edge from dark to a near-clipped sky the
/// difference grows with the brightness (red 0.42 against green 0.78 on the bright side, 0.09 against 0.17 on the dark
/// one, on an R5 C frame), so a pixel between them took far too little red from the bright side's difference: a neon
/// green-cyan rim. The ratio stays put (0.54 and 0.53 there). Blended in by [`GUIDE_DARK`] and [`GUIDE_BRIGHT`].
#[allow(clippy::too_many_arguments)]
fn by_ratio(mut p: [f32; 3], own: u8, green: f32, mos: &[f32], col: &[u8], g: &[f32], i: usize, bw: usize) -> [f32; 3] {
    let mut follow = |c: Option<&u8>, sites: &[usize]| {
        let (mut own_sum, mut green_sum) = (0f32, 0f32);
        for &j in sites {
            let (Some(&v), Some(&gj)) = (mos.get(j), g.get(j)) else { return };
            own_sum += v;
            green_sum += gj;
        }
        let v = green * (own_sum + RATIO_FLOOR) / (green_sum.max(0.0) + RATIO_FLOOR);
        let t =
            smoothstep(GUIDE_DARK.0, GUIDE_DARK.1, green_sum / sites.len().max(1) as f32) * (1.0 - smoothstep(GUIDE_BRIGHT.0, GUIDE_BRIGHT.1, green));
        if let Some(slot) = c.and_then(|&c| p.get_mut(c as usize))
            && v.is_finite()
        {
            *slot += (v - *slot) * t;
        }
    };
    if own == 1 {
        follow(col.get(i + 1), &[i - 1, i + 1]);
        follow(col.get(i + bw), &[i - bw, i + bw]);
    } else {
        follow(Some(&(2 - own)), &[i - bw - 1, i - bw + 1, i + bw - 1, i + bw + 1]);
    }
    p
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
    use super::{CLIP, RATIO_REACH, REACH, ahd_with};
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
        // out of reach (red and blue follow green 6 px around, and the homogeneity reads 3 more): plain AHD, bit for bit
        let (w, h) = (n.width, n.height);
        let r = RATIO_REACH + 3;
        let mut far = 0;
        for y in 0..h {
            for x in 0..w {
                let reach = (y.saturating_sub(r)..(y + r + 1).min(h))
                    .any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| n.data[yy * w + xx] >= CLIP));
                if !reach {
                    far += 1;
                    assert_eq!(aware.get(x, y), plain.get(x, y), "({x}, {y})");
                }
            }
        }
        assert!(far > 300, "{far} pixels out of reach");
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

    /// A dark beam across a pale blue sky that brightens away from it until green, then blue clip a pixel or two from
    /// its edges (red never does), as at the edge of a clipped sky. The edges are soft, as a lens draws them, and
    /// slanted, so they cross the pattern at every phase, sky above one and below the other. `peak` 0.65: nothing clips.
    fn beam(peak: f32) -> Rgb32f {
        let (sky, dark) = ([0.38f32, 1.0, 0.82], [0.02f32, 0.045, 0.03]);
        Rgb32f::from_fn(96, 64, |x, y| {
            // distance out of the 10 px beam (negative inside)
            let d = (y as f32 + 0.5 - (30.0 + x as f32 / 8.0)).abs() - 5.0;
            let t = 0.5 + 0.5 * (d / 0.8).tanh();
            let level = peak * (0.92 + 0.08 * d.clamp(0.0, 6.0));
            std::array::from_fn(|c| (dark[c] + (sky[c] * level - dark[c]) * t).min(1.0))
        })
    }

    /// Pixels on an edge (green between 15 % and 85 % of the sky's nearby) whose red or blue ratio to green lies more
    /// than 10 % outside the range the two sides' samples span (colour neither side has: the neon rim), and the count
    /// of edge pixels.
    fn false_colour(img: &Rgb32f) -> (usize, usize) {
        let (mut bad, mut n) = (0, 0);
        for y in 2..img.height - 2 {
            for x in 2..img.width - 2 {
                let p = img.get(x, y);
                let sky = (y.saturating_sub(8)..(y + 9).min(img.height)).map(|yy| img.get(x, yy)[1]).fold(1e-3, f32::max);
                if !(0.15..0.85).contains(&(p[1] / sky)) {
                    continue;
                }
                n += 1;
                // red / green: sky 0.38 (unclipped) to 0.53 (green clipped), beam 0.44; blue / green: 0.67 to 1.0
                let (rg, bg) = (p[0] / p[1], p[2] / p[1]);
                if !(0.38 / 1.1..=0.53 * 1.1).contains(&rg) || !(0.67 / 1.1..=1.1).contains(&bg) {
                    bad += 1;
                }
            }
        }
        (bad, n)
    }

    /// Process V9: next to a clipped sky an edge's red and blue were green plus the difference of samples on either side
    /// of it, which grows with the brightness, so a pixel between them took far too little of a colour (a neon
    /// green-cyan rim). As a ratio to green they keep the colours of the two sides.
    #[test]
    fn next_to_a_clipped_sky_an_edge_keeps_its_colours() {
        let (mut before, mut total) = (0, 0);
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"] {
            let cfa = Cfa::bayer(pattern).unwrap();
            let n = mosaic_from_rgb(&beam(1.0), &cfa);
            assert!(n.data.iter().filter(|&&v| v >= CLIP).count() > 500, "{pattern}: the sky clips");
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            let (was, count) = false_colour(&ahd_with(&m, true, false));
            let (after, _) = false_colour(&ahd_with(&m, true, true));
            eprintln!("{pattern}: false colour on the edges {was} -> {after} of {count}");
            assert!(count >= 200, "{pattern}: {count} edge pixels");
            assert!(after * 40 <= count, "{pattern}: {after} of {count}");
            (before, total) = (before + was, total + count);
        }
        assert!(before * 10 >= total, "plain AHD draws the rim: {before} of {total}");
    }

    /// Small clipped glints on a dark, noisy surface (catchlights in an eye): the dark pixels around them keep their
    /// colour on average (red and blue as a ratio to a green that noise dominates would come out too high: magenta).
    /// Measured where the ratio reaches and the clamp to the samples' range ([`REACH`]) doesn't.
    #[test]
    fn dark_noise_around_glints_keeps_its_colour() {
        let img = Rgb32f::from_fn(96, 64, |x, y| if x % 16 == 7 && y % 16 == 7 { [4.0; 3] } else { [0.012, 0.025, 0.018] });
        let img = Rgb32f { data: img.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..img };
        let cfa = Cfa::bayer("RGGB").unwrap();
        let mut n = mosaic_from_rgb(&img, &cfa);
        // deterministic noise, about 0.004 of white per sample
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for v in n.data.iter_mut().filter(|v| **v < 0.5) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v += ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.014;
        }
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        let (plain, aware) = (ahd_with(&m, true, false), ahd_with(&m, true, true));
        let (w, h) = (n.width, n.height);
        let (mut sum_plain, mut sum_aware, mut count) = ([0f64; 3], [0f64; 3], 0);
        for y in 0..h {
            for x in 0..w {
                let near = |r: usize| {
                    (y.saturating_sub(r)..(y + r + 1).min(h))
                        .any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| n.data[yy * w + xx] >= CLIP))
                };
                if near(RATIO_REACH) && !near(REACH) {
                    count += 1;
                    for c in 0..3 {
                        sum_plain[c] += plain.get(x, y)[c] as f64;
                        sum_aware[c] += aware.get(x, y)[c] as f64;
                    }
                }
            }
        }
        assert!(count > 200, "{count} dark pixels near the glints");
        for c in [0, 2] {
            let ratio = sum_aware[c] / sum_plain[c];
            eprintln!("channel {c}: mean {ratio:.4} of plain AHD");
            assert!((ratio - 1.0).abs() < 0.005, "channel {c}: {ratio}");
        }
    }

    /// Glints in a textured field just below the clip (green above 0.95 of the white, the highlight rebuild's rim):
    /// red and blue keep their colour differences there, so beyond the clamp's reach and its homogeneity's (6 px)
    /// every pixel is plain AHD's, though a ratio would reach further.
    #[test]
    fn near_the_clip_red_and_blue_keep_their_differences() {
        let img = Rgb32f::from_fn(96, 64, |x, y| {
            if x % 24 == 11 && y % 24 == 11 {
                return [4.0; 3];
            }
            let (fx, fy) = (x as f32, y as f32);
            let t = (fx * 0.9).sin() * (fy * 0.7).cos();
            [0.55 + 0.08 * t, 0.97 + 0.012 * t, 0.7 - 0.06 * t]
        });
        let img = Rgb32f { data: img.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..img };
        let cfa = Cfa::bayer("RGGB").unwrap();
        let n = mosaic_from_rgb(&img, &cfa);
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        let (plain, aware) = (ahd_with(&m, true, false), ahd_with(&m, true, true));
        let (w, h) = (n.width, n.height);
        let mut beyond = 0;
        for y in 0..h {
            for x in 0..w {
                let near = |r: usize| {
                    (y.saturating_sub(r)..(y + r + 1).min(h))
                        .any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| n.data[yy * w + xx] >= CLIP))
                };
                if near(RATIO_REACH + 3) && !near(2 * REACH) {
                    beyond += 1;
                    assert_eq!(aware.get(x, y), plain.get(x, y), "({x}, {y})");
                }
            }
        }
        assert!(beyond > 100, "{beyond} pixels between the reaches");
    }
}
