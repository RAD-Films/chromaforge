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
//! On every edge, clipped or not, [`super::Method::AhdBlend`]'s red and blue also follow green's shape: green times
//! their ratio to green at the samples they are interpolated from, not green plus their difference to it
//! ([`by_ratio`], the rule and its tunables in `edge.rs`). From a dark bridge to a bright sky that difference grows with
//! the brightness, so a pixel on the edge took too much or too little red or blue from the bright side's difference: a
//! neon fringe. Where the pixel's green and its sites' green agree (flat and smoothly shaded areas) the rule doesn't
//! run and the candidates are bit for bit as before.
//!
//! Work is done in overlapping tiles (bounded memory, rayon-parallel), one band of tiles at a time written
//! straight into the output. Each step runs only where the tile's output depends on it: the choice reads the
//! homogeneity 1 pixel around the tile, which reads Lab 2 pixels around, the candidates read the greens 3 pixels around
//! (the ratio's outer sites 3 more), and those read the mosaic 2 pixels around, inside the `MARGIN`.

use super::Mosaic;
use super::edge::{self, Follow, SensorNoise};
use crate::Rgb32f;
use rayon::prelude::*;

const TILE: usize = 128;
const MARGIN: usize = 7;
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
/// Samples' worth of one directional green at a site, for the ratio's noise ([`Follow::trust`]): the mean of two greens
/// plus a quarter of the site colour's second difference has about one sample's variance.
const SITE_SAMPLES: f32 = 1.0;

/// `blend`: blend the two candidates where their homogeneity sums are close ([`BLEND`], process V9's
/// [`super::Method::AhdBlend`]), keep them within their samples' range near clipped ones, and let red and blue follow
/// green across edges for a file with this `noise` ([`by_ratio`]); false: the more homogeneous one outright, the
/// average on ties (V1-V8), and neither rule.
pub(crate) fn ahd(m: &Mosaic, blend: bool, noise: SensorNoise) -> Rgb32f {
    ahd_with(m, blend, blend, blend.then_some(noise))
}

/// [`ahd`], with the candidates kept within their samples' range near clipped ones (`clip_aware`) or not, and red and
/// blue following green across edges for a file with this noise ([`by_ratio`]) or not.
fn ahd_with(m: &Mosaic, blend: bool, clip_aware: bool, follow: Option<SensorNoise>) -> Rgb32f {
    let rule = follow.map(Follow::new);
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
                (x0, tw, tile(m, x0, y0, tw, th, blend, clip_aware, rule.as_ref()))
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

#[allow(clippy::too_many_arguments)]
fn tile(m: &Mosaic, x0: usize, y0: usize, tw: usize, th: usize, blend: bool, clip_aware: bool, rule: Option<&Follow>) -> Vec<[f32; 3]> {
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
    // and the sites the ratio leaves out: a clipped sample or one beside it, whose green the clip understates
    let touched = if rule.is_some() && near.is_some() { Some(clip_touched(&mos, bw)) } else { None };
    // and the pixels within [`edge::NEAR_CLIP`] of one, where the ratio runs on every pixel (with the dark guard)
    let zone = if rule.is_some() { near_clipped_wide(m, &mos, (ox, oy), bw, bh) } else { None };
    let sites = Sites::new(bw);
    // 1. directional greens (down to 2 px from the buffer's edge: the ratio's outer sites read them 3 px around)
    let mut gh = vec![0f32; n];
    let mut gv = vec![0f32; n];
    for i in inner(2) {
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
            let p = match rule {
                Some(rule) => {
                    let near_clip = zone.as_ref().is_some_and(|z| z.get(i).copied().unwrap_or(false));
                    if near_clip || on_edge(rule, own, gvv, g, i, bw) {
                        by_ratio(p, own, gvv, &mos, &col, g, i, &sites, rule, touched.as_deref(), near_clip)
                    } else {
                        p
                    }
                }
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
/// ([`CLIP`]) within [`edge::NEAR_CLIP`] (a square), or `None` when none has. The candidates are completed down to
/// `MARGIN - 2` px from the buffer's edge, and their square reaches past it: the mosaic is read that much beyond the
/// buffer, so that a tile's edge changes nothing.
fn near_clipped_wide(m: &Mosaic, mos: &[f32], at: (isize, isize), bw: usize, bh: usize) -> Option<Vec<bool>> {
    let reach = edge::NEAR_CLIP;
    let extra = reach.saturating_sub(MARGIN - 2);
    let (ew, eh) = (bw + 2 * extra, bh + 2 * extra);
    let (ex, ey) = (at.0 - extra as isize, at.1 - extra as isize);
    let clipped_at = |x: usize, y: usize| m.get(ex + x as isize, ey + y as isize) >= CLIP;
    // (the ring outside the buffer first: a tile with no clipped sample in or around it is the rule)
    let ring = (0..eh).any(|y| {
        if y < extra || y + extra >= eh { (0..ew).any(|x| clipped_at(x, y)) } else { (0..extra).chain(ew - extra..ew).any(|x| clipped_at(x, y)) }
    });
    if !ring && !mos.iter().any(|&v| v >= CLIP) {
        return None;
    }
    let clipped: Vec<bool> = (0..eh).flat_map(|y| (0..ew).map(move |x| (x, y))).map(|(x, y)| clipped_at(x, y)).collect();
    let mut rows = vec![false; ew * eh];
    for (src, dst) in clipped.chunks_exact(ew).zip(rows.chunks_exact_mut(ew)) {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src.get(x.saturating_sub(reach)..(x + reach + 1).min(ew)).is_some_and(|s| s.contains(&true));
        }
    }
    let mut near = vec![false; bw * bh];
    for (y, row) in near.chunks_exact_mut(bw).enumerate() {
        let yy = y + extra;
        for (x, n) in row.iter_mut().enumerate() {
            let xx = x + extra;
            *n = (yy.saturating_sub(reach)..(yy + reach + 1).min(eh)).any(|r| rows.get(r * ew + xx).copied().unwrap_or(false));
        }
    }
    Some(near)
}

/// Which samples of the tile buffer `mos` (`bw` wide) are clipped ([`CLIP`]) or have a clipped sample beside them
/// (left, right, above, below: the greens a site's directional green is interpolated from).
fn clip_touched(mos: &[f32], bw: usize) -> Vec<bool> {
    let clipped = |j: Option<usize>| j.and_then(|j| mos.get(j)).is_some_and(|&v| v >= CLIP);
    (0..mos.len())
        .map(|j| {
            clipped(Some(j)) || clipped(j.checked_sub(1)) || clipped(j.checked_add(1)) || clipped(j.checked_sub(bw)) || clipped(j.checked_add(bw))
        })
        .collect()
}

/// Offsets in a tile buffer `bw` wide of the sites a pixel's missing colour is interpolated from, `(nearest, outer)`
/// ([`edge::SUPPORT_OUTER`]): at a green pixel, the colour of its row neighbours (`row`) and of its column neighbours
/// (`column`); at a red or blue pixel, the other one, on the diagonals (`diagonal`). Each set lies symmetrically
/// around the pixel, so on a linear ramp its mean green is the pixel's.
struct Sites {
    row: ([isize; 2], [isize; 4]),
    column: ([isize; 2], [isize; 4]),
    diagonal: ([isize; 4], [isize; 8]),
}

impl Sites {
    fn new(bw: usize) -> Sites {
        let w = bw as isize;
        Sites {
            row: ([-1, 1], [-1 - 2 * w, 1 - 2 * w, -1 + 2 * w, 1 + 2 * w]),
            column: ([-w, w], [-w - 2, -w + 2, w - 2, w + 2]),
            diagonal: ([-w - 1, -w + 1, w - 1, w + 1], [-3 * w - 1, -3 * w + 1, 3 * w - 1, 3 * w + 1, -w - 3, -w + 3, w - 3, w + 3]),
        }
    }
}

/// Whether the pixel at flat index `i` of the tile buffer (own colour `own`, green `green` from the candidate's green
/// plane `g`) may lie on an edge for [`by_ratio`] ([`Follow::may_step`] to the nearest sites of a colour it lacks): a
/// cheap first test, false for nearly every pixel of a flat or smoothly shaded area. `i` lies at least `MARGIN - 2`
/// from the buffer's edge.
#[inline(always)]
fn on_edge(rule: &Follow, own: u8, green: f32, g: &[f32], i: usize, bw: usize) -> bool {
    let step = |a: f32, b: f32| rule.may_step(green, a.min(b), a.max(b));
    if own == 1 {
        step(g[i - 1], g[i + 1]) || step(g[i - bw], g[i + bw])
    } else {
        let (a, b, c, d) = (g[i - bw - 1], g[i - bw + 1], g[i + bw - 1], g[i + bw + 1]);
        rule.may_step(green, a.min(b).min(c.min(d)), a.max(b).max(c.max(d)))
    }
}

/// `p` (the candidate at flat index `i` of the tile buffer, own colour `own`, green `green` from the candidate's green
/// plane `g`) with each interpolated red and blue moved toward green times its ratio to green at the sites it is
/// interpolated from (and the next ones out), instead of green plus its difference to it. Across an edge from a dark
/// bridge to a bright sky the difference grows with the brightness (red 0.42 against green 0.78 on the bright side,
/// 0.09 against 0.17 on the dark one, on an R5 C frame), so a pixel between them took far too little red from the
/// bright side's difference: a neon green-cyan fringe. The ratio stays put (0.54 and 0.53 there). How far: the edge
/// gate between `green` and the nearest sites' greens ([`Follow::edge`]; 0 leaves `p` exactly as it is), the
/// trust in the ratio given the sites' green noise ([`Follow::trust`]) and the near-clip fade ([`edge::fade`]).
///
/// The sums leave out the sites `touched` marks ([`clip_touched`]): a clipped sample reads lower than the light was,
/// and so does a green interpolated from one, so a ratio to it overstates the colour. Next to a clipped sky that made
/// red and blue too high on the bright side of dark bars, a magenta line a -3 EV render showed; the outer sites, 3 px
/// out, reach into the clipping first. When every nearest site is touched the colour difference stays.
///
/// Within [`edge::NEAR_CLIP`] of a clipped sample (`near_clip`) the ratio runs on every pixel, as far as the larger of
/// the edge rule's weight and the dark guard ([`edge::NEAR_CLIP_DARK`]) allow: there the colour difference itself reads
/// against the clipping and leaves red short (cyan), and at higher ISO the noise allowance switched the ratio off.
#[allow(clippy::too_many_arguments)]
fn by_ratio(
    mut p: [f32; 3],
    own: u8,
    green: f32,
    mos: &[f32],
    col: &[u8],
    g: &[f32],
    i: usize,
    sites: &Sites,
    rule: &Follow,
    touched: Option<&[bool]>,
    near_clip: bool,
) -> [f32; 3] {
    let fade = if green <= edge::GUIDE_BRIGHT.0 { 1.0 } else { edge::fade(green) };
    if fade.is_nan() || fade <= 0.0 {
        return p;
    }
    // green's range over some sites (cheap: most pixels stop here)
    let range = |offsets: &[isize]| {
        offsets.iter().try_fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &o| {
            let gj = *g.get(i.wrapping_add_signed(o))?;
            Some((lo.min(gj), hi.max(gj)))
        })
    };
    // the colour's and green's sums over some sites and how many they are, leaving out the sites next to a clipped
    // sample (`None` past the buffer, which the margin rules out)
    let sum = |offsets: &[isize]| {
        offsets.iter().try_fold((0f32, 0f32, 0f32), |(own, green, n), &o| {
            let j = i.wrapping_add_signed(o);
            if touched.is_some_and(|t| t.get(j).copied().unwrap_or(true)) {
                return Some((own, green, n));
            }
            Some((own + *mos.get(j)?, green + *g.get(j)?, n + 1.0))
        })
    };
    let mut follow = |c: Option<&u8>, near: &[isize], outer: &[isize]| {
        let Some((lo, hi)) = range(near) else { return };
        let gate = if rule.may_step(green, lo, hi) { rule.edge(green, lo, hi, SITE_SAMPLES) * fade } else { 0.0 };
        if !near_clip && (gate.is_nan() || gate <= 0.0) {
            return;
        }
        // (every nearest site next to a clipped sample: the colour difference stays)
        let (Some((own_near, green_near, n)), Some((own_outer, green_outer, m))) = (sum(near), sum(outer)) else { return };
        if n <= 0.0 {
            return;
        }
        let w = edge::SUPPORT_OUTER;
        let total = n + w * m;
        let (own_mean, guide) = ((own_near + w * own_outer) / total, (green_near + w * green_outer) / total);
        let t = gate * rule.trust(guide, edge::samples(total, n + w * w * m, SITE_SAMPLES));
        // (near clipped samples, at least as far as the sites' green allows, edge or not)
        let t = if near_clip { t.max(fade * edge::smoothstep(edge::NEAR_CLIP_DARK.0, edge::NEAR_CLIP_DARK.1, guide)) } else { t };
        let v = edge::ratio(green, own_mean, guide);
        if t > 0.0
            && v.is_finite()
            && let Some(slot) = c.and_then(|&c| p.get_mut(c as usize))
        {
            *slot += (v - *slot) * t;
        }
    };
    if own == 1 {
        follow(col.get(i + 1), &sites.row.0, &sites.row.1);
        follow(col.get(i.wrapping_add_signed(sites.column.0[1])), &sites.column.0, &sites.column.1);
    } else {
        follow(Some(&(2 - own)), &sites.diagonal.0, &sites.diagonal.1);
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

    use super::super::{Mosaic, SensorNoise, mosaic_from_rgb};
    use super::{CLIP, REACH, ahd_with};
    use crate::{Cfa, Rgb32f};

    const PATTERNS: [&str; 4] = ["RGGB", "BGGR", "GRBG", "GBRG"];

    /// A 14-bit file at ISO 100: the noise these noise-free scenes are developed for.
    fn clean() -> SensorNoise {
        SensorNoise::new(16_383.0 - 512.0, Some(100))
    }

    /// Process V9 as it develops now.
    fn v9(m: &Mosaic, noise: SensorNoise) -> Rgb32f {
        ahd_with(m, true, true, Some(noise))
    }

    /// Process V9 without the edge rule: as it developed before red and blue followed green across every edge (the same
    /// as before for a frame without a clipped sample).
    fn without_rule(m: &Mosaic) -> Rgb32f {
        ahd_with(m, true, true, None)
    }

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

    /// Whether `(x, y)` lies within `r` px (a square) of a clipped sample.
    fn near_clip(n: &[f32], w: usize, h: usize, x: usize, y: usize, r: usize) -> bool {
        (y.saturating_sub(r)..(y + r + 1).min(h)).any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| n[yy * w + xx] >= CLIP))
    }

    /// Process V9: next to clipped samples no interpolated channel goes past its colour's samples around it (plain AHD
    /// does here: a black speck or a neon pixel), and out of the clamp's reach (its 3 px and the homogeneity's 3 more)
    /// the clamp changes nothing.
    #[test]
    fn near_clipped_samples_the_colours_stay_within_their_samples() {
        let (img, cfa) = branches(16.0);
        let n = mosaic_from_rgb(&img, &cfa);
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        let (plain, aware) = (ahd_with(&m, true, false, None), v9(&m, clean()));
        let (before, count) = overshoot(&plain, &n.data, &cfa, n.width, n.height);
        let (after, _) = overshoot(&aware, &n.data, &cfa, n.width, n.height);
        assert!(count > 500, "{count} pixels near clip");
        assert!(before > 0.05, "plain AHD overshoots its samples here: {before}");
        assert!(after <= 1e-6, "clip-aware: {after}");
        assert!(aware.data.iter().all(|p| p.iter().all(|v| v.is_finite() && *v >= 0.0)));
        let unclamped = ahd_with(&m, true, false, Some(clean()));
        let (w, h) = (n.width, n.height);
        let mut far = 0;
        for y in 0..h {
            for x in 0..w {
                if !near_clip(&n.data, w, h, x, y, 2 * REACH) {
                    far += 1;
                    assert_eq!(aware.get(x, y), unclamped.get(x, y), "({x}, {y})");
                }
            }
        }
        assert!(far > 300, "{far} pixels out of reach");
    }

    /// A frame without a clipped sample is never clamped.
    #[test]
    fn an_unclipped_frame_is_not_clamped() {
        let (img, cfa) = branches(0.95);
        let n = mosaic_from_rgb(&img, &cfa);
        assert!(n.data.iter().all(|&v| v < CLIP));
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        assert_eq!(v9(&m, clean()).data, ahd_with(&m, true, false, Some(clean())).data);
    }

    /// A dark beam across a pale blue sky that brightens away from it until green, then blue clip a pixel or two from
    /// its edges (red never does), as at the edge of a clipped sky. The edges are soft, as a lens draws them, and
    /// slanted, so they cross the pattern at every phase, sky above one and below the other.
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

    /// Pixels on an edge (green between 15 % and 85 % of the brightest green within `reach` rows and columns) whose red
    /// or blue ratio to green lies more than 10 % outside the range the two sides span (`rg`, `bg`: colour neither side
    /// has, a fringe), and the count of edge pixels.
    fn false_colour(img: &Rgb32f, reach: usize, rg: (f32, f32), bg: (f32, f32)) -> (usize, usize) {
        let (mut bad, mut n) = (0, 0);
        let (w, h) = (img.width, img.height);
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let p = img.get(x, y);
                let bright = (y.saturating_sub(reach)..(y + reach + 1).min(h))
                    .flat_map(|yy| (x.saturating_sub(reach)..(x + reach + 1).min(w)).map(move |xx| (xx, yy)))
                    .map(|(xx, yy)| img.get(xx, yy)[1])
                    .fold(1e-3, f32::max);
                if !(0.15..0.85).contains(&(p[1] / bright)) {
                    continue;
                }
                n += 1;
                let (r, b) = (p[0] / p[1], p[2] / p[1]);
                if !(rg.0 / 1.1..=rg.1 * 1.1).contains(&r) || !(bg.0 / 1.1..=bg.1 * 1.1).contains(&b) {
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
        for pattern in PATTERNS {
            let cfa = Cfa::bayer(pattern).unwrap();
            let n = mosaic_from_rgb(&beam(1.0), &cfa);
            assert!(n.data.iter().filter(|&&v| v >= CLIP).count() > 500, "{pattern}: the sky clips");
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            // red / green: sky 0.38 (unclipped) to 0.53 (green clipped), beam 0.44; blue / green: 0.67 to 1.0
            let measure = |img: &Rgb32f| false_colour(img, 8, (0.38, 0.53), (0.67, 1.0));
            let (was, count) = measure(&ahd_with(&m, true, false, None));
            let (after, _) = measure(&v9(&m, clean()));
            eprintln!("{pattern}: false colour on the edges {was} -> {after} of {count}");
            assert!(count >= 200, "{pattern}: {count} edge pixels");
            assert!(after * 40 <= count, "{pattern}: {after} of {count}");
            (before, total) = (before + was, total + count);
        }
        assert!(before * 10 >= total, "plain AHD draws the rim: {before} of {total}");
    }

    /// A dark bridge's girders against a bright sky that is nowhere near its clip (the bridge of test frame 6719):
    /// soft edges as a lens draws them, one slanted and one steep, so they cross the pattern at every phase. Camera RGB:
    /// sky red / green 0.45 and blue / green 0.8, girders 0.5 and 0.75.
    fn bridge(w: usize, h: usize) -> Rgb32f {
        let (sky, dark) = ([0.27f32, 0.6, 0.48], [0.02f32, 0.04, 0.03]);
        Rgb32f::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let girder = (fy - (0.3 * h as f32 + fx / 6.0)).abs() - 3.0;
            let post = (fx - (0.6 * w as f32 + fy / 5.0)).abs() - 2.5;
            let t = 0.5 + 0.5 * (girder.min(post) / 0.7).tanh();
            std::array::from_fn(|c| dark[c] + (sky[c] - dark[c]) * t)
        })
    }

    /// Process V9: on a bright edge that clips nowhere, red and blue as green plus their difference to green at their
    /// sites gave the edge's pixels colour neither side has (a neon fringe at full size); as a ratio to green they keep
    /// the sides' colours. Without the edge rule (process V9 before this change) the fringe is there.
    #[test]
    fn a_bright_unclipped_edge_keeps_its_colours() {
        let (mut before, mut after, mut total) = (0, 0, 0);
        for pattern in PATTERNS {
            let cfa = Cfa::bayer(pattern).unwrap();
            let n = mosaic_from_rgb(&bridge(96, 64), &cfa);
            assert!(n.data.iter().all(|&v| v < 0.7), "{pattern}: nothing near the clip");
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            let measure = |img: &Rgb32f| false_colour(img, 6, (0.45, 0.5), (0.75, 0.8));
            let (was, count) = measure(&without_rule(&m));
            let (now, _) = measure(&v9(&m, clean()));
            eprintln!("{pattern}: fringe on the edges {was} -> {now} of {count}");
            assert!(count >= 150, "{pattern}: {count} edge pixels");
            (before, after, total) = (before + was, after + now, total + count);
        }
        assert!(before * 5 >= total, "without the rule the edge has a fringe: {before} of {total}");
        assert!(after * 30 <= total, "with it: {after} of {total}");
    }

    /// Dark bars against a sky two stops over the clip in green (blue clips too, red doesn't): the ring next to clipped
    /// samples, where the bars' edges climb toward the clipped sky, that a -3 EV render shows. Soft edges, one bar
    /// slanted and one steep, every pattern phase. Camera RGB: sky red / green 0.45, blue / green 0.75.
    fn ring_scene(w: usize, h: usize) -> Rgb32f {
        let (sky, dark) = ([0.45f32, 1.0, 0.75], [0.02f32, 0.04, 0.03]);
        Rgb32f::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let bar = (fy - (0.35 * h as f32 + fx / 7.0)).abs() - 2.5;
            let post = (fx - (0.65 * w as f32 + fy / 6.0)).abs() - 2.0;
            let d = bar.min(post);
            let t = 0.5 + 0.5 * (d / 0.7).tanh();
            std::array::from_fn(|c| dark[c] + (sky[c] * 2.0 - dark[c]) * t)
        })
    }

    /// Over the pixels of `out` within 5 px of a clipped sample of `n` whose true colour (`truth`) is clipped in no
    /// channel: how many have a red or blue ratio to green more than 10 % outside the range the bars and the sky span
    /// (colour neither has), how many of those are magenta (red and blue both too high against green), and the count.
    fn ring_error(out: &Rgb32f, truth: &Rgb32f, n: &[f32]) -> (usize, usize, usize) {
        let (w, h) = (out.width, out.height);
        let (mut bad, mut magenta, mut count) = (0, 0, 0);
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let t = truth.get(x, y);
                if t.iter().any(|&v| v >= 0.95) || !near_clip(n, w, h, x, y, 5) {
                    continue;
                }
                count += 1;
                let p = out.get(x, y);
                // red / green: sky 0.45, bars 0.5; blue / green 0.75 on both: any mix of the two lies between
                let (rg, bg) = (p[0] / p[1], p[2] / p[1]);
                if !(0.45 / 1.1..=0.5 * 1.1).contains(&rg) || !(0.75 / 1.1..=0.75 * 1.1).contains(&bg) {
                    bad += 1;
                    if rg > 0.5 * 1.1 && bg > 0.75 * 1.1 {
                        magenta += 1;
                    }
                }
            }
        }
        (bad, magenta, count)
    }

    /// A sky whose green and blue clip and whose red doesn't, and below it a mid-toned, textured surface (foliage, a wall)
    /// with, if `bar`, a dark bar 2.5-4.5 px out: the 0-2 and 2-5 px rings next to clipped samples that a -3 EV render
    /// shows. `level`: the surface's green; `amp`, `freq`: its texture (red varies most, blue least, so its colour moves
    /// with it). Soft, slanted edges, every pattern phase. Camera RGB: sky red / green 0.5, blue / green 1.45, at 2.5 x
    /// the white in green; surface red / green 0.62, blue / green 0.8.
    fn textured_ring_scene(w: usize, h: usize, level: f32, amp: f32, freq: f32, bar: bool) -> Rgb32f {
        let sky = [0.5f32, 1.0, 1.45].map(|v| v * 2.5);
        Rgb32f::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            // distance below the sky's edge
            let d = fy - (0.3 * h as f32 + fx / 5.0);
            let t = (fx * freq).sin() * (fy * freq * 0.85).cos();
            let surface = [level * 0.62 * (1.0 + amp * 1.2 * t), level * (1.0 + amp * t), level * 0.8 * (1.0 + amp * 0.7 * t)];
            let in_bar = if bar { 0.5 + 0.5 * ((1.0 - (d - 3.5).abs()) / 0.6).tanh() } else { 0.0 };
            let below = 0.5 + 0.5 * (d / 0.7).tanh();
            std::array::from_fn(|c| {
                let ground = surface[c] + ([0.03, 0.05, 0.04][c] - surface[c]) * in_bar;
                sky[c] + (ground - sky[c]) * below
            })
        })
    }

    /// Over the pixels of `out` between `ring.0` (exclusive) and `ring.1` px (a square) from a clipped sample of `n`
    /// whose true colour (`truth`) clips in no channel: how many have red or blue against green more than 10 % off the
    /// truth's, how many of those are cyan (red short, the larger error) and magenta (red and blue both over), the mean
    /// error (of red and blue against green, log) and the count.
    fn ring_against_truth(out: &Rgb32f, truth: &Rgb32f, n: &[f32], ring: (usize, usize)) -> [f64; 5] {
        let (w, h) = (out.width, out.height);
        let mut r = [0f64; 5];
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                let t = truth.get(x, y);
                if t.iter().any(|&v| v >= 0.95) || near_clip(n, w, h, x, y, ring.0) || !near_clip(n, w, h, x, y, ring.1) {
                    continue;
                }
                let p = out.get(x, y);
                let er = ((p[0] / p[1]) / (t[0] / t[1])).ln();
                let eb = ((p[2] / p[1]) / (t[2] / t[1])).ln();
                if !(er.is_finite() && eb.is_finite()) {
                    continue;
                }
                r[4] += 1.0;
                r[3] += f64::from(er.abs() + eb.abs()) / 2.0;
                if er.abs() > 0.1 || eb.abs() > 0.1 {
                    r[0] += 1.0;
                    if er < -0.1 && er.abs() >= eb.abs() {
                        r[1] += 1.0;
                    }
                    if er > 0.05 && eb > 0.05 && er + eb > 0.1 {
                        r[2] += 1.0;
                    }
                }
            }
        }
        r
    }

    /// The textured ring scene at ISO `iso` (noise of that model, deterministic), developed by `develop` for each
    /// pattern: [`ring_against_truth`] summed over the patterns, for the 0-2 and 2-5 px rings.
    fn textured_ring(level: f32, amp: f32, freq: f32, bar: bool, iso: u32, develop: &dyn Fn(&Mosaic, SensorNoise) -> Rgb32f) -> [[f64; 5]; 2] {
        let truth = textured_ring_scene(128, 96, level, amp, freq, bar);
        let clipped = Rgb32f { data: truth.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..truth.clone() };
        let noise = SensorNoise::new(16_383.0 - 512.0, Some(iso));
        let mut sums = [[0f64; 5]; 2];
        for pattern in PATTERNS {
            let cfa = Cfa::bayer(pattern).unwrap();
            let mut n = mosaic_from_rgb(&clipped, &cfa);
            add_noise(&mut n.data, noise, 0x9e37_79b9_7f4a_7c15 ^ u64::from(iso));
            for v in n.data.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            let out = develop(&m, noise);
            for (sum, ring) in sums.iter_mut().zip([(0, 2), (2, 5)]) {
                for (s, v) in sum.iter_mut().zip(ring_against_truth(&out, &truth, &n.data, ring)) {
                    *s += v;
                }
            }
        }
        sums
    }

    /// Process V9 within 0-5 px of clipped samples at a higher ISO (2000): a dark bar 2.5-4.5 px below a sky whose green
    /// and blue clip, and a textured mid-toned surface right against it. The colour difference, where the edge gate's
    /// noise allowance or the noise weight left it in place, read against the clipping: red came out short (cyan) in
    /// the 0-2 px ring and the texture's colour off in the 2-5 px ring. Measured against the rule switched off, which
    /// draws that colour everywhere there.
    #[test]
    fn next_to_clipped_samples_texture_keeps_its_colour() {
        let develop = |m: &Mosaic, n: SensorNoise| v9(m, n);
        let without = |m: &Mosaic, _: SensorNoise| without_rule(m);
        // the bar: cyan in the 0-2 px ring
        let (was, now) = (textured_ring(0.15, 0.1, 0.6, true, 2000, &without), textured_ring(0.15, 0.1, 0.6, true, 2000, &develop));
        eprintln!("bar, 0-2 px: cyan {} -> {}, off colour {} -> {} of {}", was[0][1], now[0][1], was[0][0], now[0][0], now[0][4]);
        assert!(now[0][4] >= 800.0, "{} pixels in the ring", now[0][4]);
        assert!(now[0][1] * 3.0 <= was[0][1], "cyan in the 0-2 px ring: {} (without the rule {})", now[0][1], was[0][1]);
        // the texture: its colour in the 2-5 px ring
        let (was, now) = (textured_ring(0.45, 0.3, 1.3, false, 2000, &without), textured_ring(0.45, 0.3, 1.3, false, 2000, &develop));
        let (err_was, err_now) = (was[1][3] / was[1][4], now[1][3] / now[1][4]);
        eprintln!("texture, 2-5 px: error {err_was:.4} -> {err_now:.4}, cyan {} -> {} of {}", was[1][1], now[1][1], now[1][4]);
        assert!(now[1][4] >= 1500.0, "{} pixels in the ring", now[1][4]);
        assert!(err_now <= 0.8 * err_was, "the texture's colour error in the 2-5 px ring: {err_now} (without the rule {err_was})");
        assert!(now[1][1] * 2.0 <= was[1][1], "cyan in the 2-5 px ring: {} (without the rule {})", now[1][1], was[1][1]);
    }

    /// Process V9 next to clipped samples, at full size: on the bright side of a dark bar against a clipped sky (the
    /// ring a -3 EV render shows), red and blue keep the sky's colour. A ratio to a green read from clipped greens,
    /// which reads too low, made them too high there: a magenta line.
    #[test]
    fn next_to_clipped_samples_the_ring_keeps_its_colour() {
        let truth = ring_scene(112, 80);
        let clipped = Rgb32f { data: truth.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..truth.clone() };
        let (mut plain_bad, mut plain_magenta, mut bad, mut magenta, mut total) = (0, 0, 0, 0, 0);
        for pattern in PATTERNS {
            let cfa = Cfa::bayer(pattern).unwrap();
            let n = mosaic_from_rgb(&clipped, &cfa);
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            let (was, was_magenta, count) = ring_error(&without_rule(&m), &truth, &n.data);
            let (now, now_magenta, _) = ring_error(&v9(&m, clean()), &truth, &n.data);
            eprintln!("{pattern}: off colour in the ring {was} ({was_magenta} magenta) -> {now} ({now_magenta} magenta) of {count}");
            (plain_bad, plain_magenta, bad, magenta, total) =
                (plain_bad + was, plain_magenta + was_magenta, bad + now, magenta + now_magenta, total + count);
        }
        eprintln!("ring: {plain_bad} ({plain_magenta} magenta) -> {bad} ({magenta} magenta) of {total}");
        assert!(total >= 200, "{total} ring pixels");
        assert!(magenta * 10 <= total, "magenta in the ring: {magenta} of {total}");
        assert!(magenta <= plain_magenta, "the rule adds magenta in the ring: {plain_magenta} -> {magenta}");
        assert!(bad <= plain_bad, "the rule adds off colour in the ring: {plain_bad} -> {bad} of {total}");
    }

    /// Flat and smoothly shaded areas (a flat colour, linear ramps, a slow colour wave, a bright gradient with mild noise)
    /// develop bit for bit as without the edge rule: their greens never step far enough for the gate.
    #[test]
    fn flat_and_smooth_areas_develop_as_before() {
        let (w, h) = (80, 64);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let gradient = Rgb32f::from_fn(w, h, |x, y| [0.25 + 0.002 * x as f32, 0.5 + 0.002 * y as f32, 0.35]);
        let noisy = Rgb32f { data: gradient.data.iter().map(|p| p.map(|v| v * (1.0 + 0.01 * uniform()))).collect(), ..gradient };
        let scenes = [
            ("flat", Rgb32f::filled(w, h, [0.21, 0.47, 0.33])),
            // (green steps at most 0.04 stops a pixel)
            ("ramps", Rgb32f::from_fn(w, h, |x, y| [0.05 + 0.004 * x as f32, 0.15 + 0.004 * y as f32, 0.1 + 0.002 * (x + y) as f32])),
            (
                "wave",
                Rgb32f::from_fn(w, h, |x, y| {
                    let t = (x as f32 * 0.08).sin() * (y as f32 * 0.06).cos();
                    [0.2 + 0.05 * t, 0.4 + 0.08 * t, 0.3 - 0.04 * t]
                }),
            ),
            ("noisy gradient", noisy),
        ];
        for (name, img) in &scenes {
            for pattern in PATTERNS {
                let cfa = Cfa::bayer(pattern).unwrap();
                let n = mosaic_from_rgb(img, &cfa);
                let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
                for noise in [clean(), SensorNoise::new(4095.0 - 256.0, Some(3200)), SensorNoise::default()] {
                    let (now, was) = (v9(&m, noise), without_rule(&m));
                    let moved: Vec<(usize, usize)> =
                        (0..now.data.len()).filter(|&i| now.data[i] != was.data[i]).map(|i| (i % now.width, i / now.width)).collect();
                    assert!(moved.is_empty(), "{name} {pattern} {noise:?}: {} pixels moved, {:?}", moved.len(), &moved[..moved.len().min(12)]);
                }
            }
        }
    }

    /// Gaussian noise of the given model on a mosaic (deterministic; sum of four uniforms).
    fn add_noise(data: &mut [f32], noise: SensorNoise, seed: u64) {
        let mut s = seed | 1;
        let mut uniform = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        for v in data.iter_mut() {
            let g = (0..4).map(|_| uniform()).sum::<f32>() * 3f32.sqrt();
            *v += g * noise.variance(*v).sqrt();
        }
    }

    /// A dark, noisy scene shot at a high ISO (girders and textured shadows at 1-8 % of the white, sky-less): the edge
    /// rule moves no channel's mean against green's. A ratio to a green that noise dominates comes out too high on
    /// average (magenta shadows); the noise model keeps the rule out of them.
    #[test]
    fn a_dark_noisy_frame_keeps_its_colour() {
        let (w, h) = (128, 96);
        let img = Rgb32f::from_fn(w, h, |x, y| {
            let t = 0.5 + 0.5 * ((x as f32 * 0.3).sin() * (y as f32 * 0.2).cos());
            let bar = (x / 9) % 4 == 0 || (y / 11) % 5 == 0;
            let l = if bar { 0.012 } else { 0.03 + 0.05 * t };
            [l * 0.5, l, l * 0.75]
        });
        let means = |img: &Rgb32f| {
            let mut s = [0f64; 3];
            for p in &img.data {
                for c in 0..3 {
                    s[c] += f64::from(p[c]);
                }
            }
            [s[0] / s[1], s[2] / s[1]]
        };
        let high = SensorNoise::new(16_383.0 - 512.0, Some(6400));
        for pattern in ["RGGB", "GBRG"] {
            let cfa = Cfa::bayer(pattern).unwrap();
            let mut n = mosaic_from_rgb(&img, &cfa);
            add_noise(&mut n.data, high, 0x9e37_79b9_7f4a_7c15);
            let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
            let (was, now) = (means(&without_rule(&m)), means(&v9(&m, high)));
            // the same mosaic told it came from ISO 100: the ratio runs in the noise
            let wrong = means(&v9(&m, clean()));
            for c in 0..2 {
                let (shift, wrong) = (now[c] / was[c] - 1.0, wrong[c] / was[c] - 1.0);
                eprintln!("{pattern}: {} / green {shift:+.5} (as if ISO 100: {wrong:+.5})", ["red", "blue"][c]);
                assert!(shift.abs() < 0.002, "{pattern} channel {c}: {shift}");
            }
        }
    }

    /// Small clipped glints on a dark, noisy surface (catchlights in an eye): the dark pixels around them keep their
    /// colour on average (red and blue as a ratio to a green that noise dominates would come out too high: magenta).
    #[test]
    fn dark_noise_around_glints_keeps_its_colour() {
        let img = Rgb32f::from_fn(96, 64, |x, y| if x % 16 == 7 && y % 16 == 7 { [4.0; 3] } else { [0.012, 0.025, 0.018] });
        let img = Rgb32f { data: img.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..img };
        let cfa = Cfa::bayer("RGGB").unwrap();
        let mut n = mosaic_from_rgb(&img, &cfa);
        // deterministic noise, about 0.004 of white per sample (an ISO 800 file's at this level)
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for v in n.data.iter_mut().filter(|v| **v < 0.5) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v += ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.014;
        }
        let m = Mosaic { w: n.width, h: n.height, data: &n.data, cfa: &cfa };
        let (plain, aware) = (without_rule(&m), v9(&m, SensorNoise::new(16_383.0 - 512.0, Some(800))));
        let (w, h) = (n.width, n.height);
        let (mut sum_plain, mut sum_aware, mut count) = ([0f64; 3], [0f64; 3], 0);
        for y in 0..h {
            for x in 0..w {
                if near_clip(&n.data, w, h, x, y, 6) && !near_clip(&n.data, w, h, x, y, REACH) {
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
            eprintln!("channel {c}: mean {ratio:.4} of V9 without the rule");
            assert!((ratio - 1.0).abs() < 0.005, "channel {c}: {ratio}");
        }
    }

    /// Glints in a textured field just below the clip (green above 0.95 of the white, the highlight rebuild's rim):
    /// red and blue keep their colour differences there (the near-clip fade), every pixel as without the rule.
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
        assert!(v9(&m, clean()).data == without_rule(&m).data);
    }
}
