//! Reduced-size development straight from the mosaic ("superpixel" binning).
//!
//! When the caller needs an image at most half (or a third, a quarter…) of the sensor size — a
//! 2560 px loupe preview or a grid thumbnail of a 24 MP raw — demosaicing the full mosaic and then
//! downscaling throws most of the work away. Instead every `k × k` block of CFA samples becomes one
//! output pixel whose channels are the means of the block's samples of that colour. Every block must
//! contain all three colours (Bayer: `k` even; X-Trans: `k` a multiple of 3). There is no
//! interpolation, so there are no demosaicing artefacts; the colour planes are offset by less than
//! one output pixel, which is invisible at these scales.
//!
//! Clipping is decided per sample, before averaging. [`RawImage::develop_binned_masked`] keeps every
//! channel the mean of its samples and says, per output pixel, which colours had a sample at or above
//! `clip` (for [`crate::highlight::reconstruct_masked`], which rebuilds them with the mean as a lower
//! bound). A specular point that clips one green sample of the 32 in an 8 × 8 block thus counts as
//! partly clipped without the block's green jumping to the clip level, which would read as a green
//! speckle once reconstructed. [`RawImage::develop_binned`] instead gives such a channel the block's
//! maximum sample of the colour, so that plain value thresholds see it as clipped.
//!
//! Process V9 ([`RawImage::develop_binned_masked_aligned`]) reads a Bayer mosaic's colour planes at the block centres,
//! and across edges red and blue follow green's shape (`follow_green`, the rule and its tunables in
//! `demosaic/edge.rs`): read between neighbouring blocks alone, a steep edge from a dark bridge or branches to a bright
//! sky took red from more of one side than green, and blue from more of the other, a violet line that a full-size
//! render doesn't have. Flat and smoothly shaded areas bin bit for bit as the aligned means.

use crate::demosaic::edge::{self, Follow, SUPPORT_OUTER};
use crate::{Normalized, RawData, RawImage, Result, Rgb32f};
use rayon::prelude::*;

impl RawImage {
    /// Can `k × k` blocks be binned (single-plane CFA data, every `k × k` window containing R, G
    /// and B; Bayer: `k` even, so that every block has the same layout)?
    pub fn can_bin(&self, k: usize) -> bool {
        let Some(cfa) = &self.cfa else { return false };
        if k < 2 || self.cpp != 1 || (self.pixel_aspect - 1.0).abs() >= 0.01 || !cfa.valid() || (cfa.is_bayer() && !k.is_multiple_of(2)) {
            return false;
        }
        // every window position relative to the pattern
        (0..cfa.height).all(|j| (0..cfa.width).all(|i| block_layout(cfa, i, j, k).1.iter().all(|&n| n > 0)))
    }

    /// Like [`RawImage::develop`] at `1/k` of the size (camera RGB, white = 1.0, `OpcodeList3` and the default crop
    /// applied, not oriented). `None` when the data cannot be binned ([`RawImage::can_bin`]) or its `OpcodeList3`
    /// holds an operation that needs every pixel ([`crate::opcodes::Opcode::binned`]); callers then demosaic.
    pub fn develop_binned(&self, k: usize, clip: f32) -> Result<Option<Rgb32f>> {
        Ok(self.bin(k, clip, true, false)?.map(|(img, _)| img))
    }

    /// Like [`RawImage::develop_binned`], with every channel the mean of its block's samples, and the
    /// clip mask: per output pixel, bits [`crate::highlight::CLIPPED_R`] / `_G` / `_B` for the colours
    /// with a sample at or above `clip` in the block. Hand both to [`crate::highlight::reconstruct_masked`].
    pub fn develop_binned_masked(&self, k: usize, clip: f32) -> Result<Option<(Rgb32f, Vec<u8>)>> {
        self.bin(k, clip, false, false)
    }

    /// [`RawImage::develop_binned_masked`] with the colour planes of a Bayer mosaic moved onto the block's centre. Its red
    /// and blue samples lie half a sensor pixel to either side of the centre, so on a hard edge one output row took its
    /// red from the dark side and its blue from the bright side: a coloured line that a full-size render doesn't have.
    /// Across edges red and blue also follow green's shape where the file's noise lets their ratio to green be trusted
    /// (`follow_green`, [`RawImage::sensor_noise`]); in flat and smoothly shaded areas the blocks are the aligned means
    /// alone.
    pub fn develop_binned_masked_aligned(&self, k: usize, clip: f32) -> Result<Option<(Rgb32f, Vec<u8>)>> {
        self.bin(k, clip, false, true)
    }

    /// The binned image and its clip mask; `keep_max`: a channel with a clipped sample takes the
    /// block's maximum sample of the colour instead of the mean.
    fn bin(&self, k: usize, clip: f32, keep_max: bool, align: bool) -> Result<Option<(Rgb32f, Vec<u8>)>> {
        self.validate()?;
        if !self.can_bin(k) {
            return Ok(None);
        }
        // `OpcodeList3` (lens shading as a gain map, lens corrections) works on the demosaiced active area: its
        // operations re-expressed for the binned image, or a full-size develop when one needs every pixel
        let Some(list3) = self.opcodes.list3.iter().map(|op| op.binned(k)).collect::<Option<Vec<_>>>() else { return Ok(None) };
        let Some(cfa) = &self.cfa else { return Ok(None) };
        let a = self.active_area;
        let whole = crate::Rect::new(0, 0, a.width, a.height);
        let crop = self.crop.clipped(a.width, a.height);
        let crop = if crop.width < k || crop.height < k { whole } else { crop };
        // bin the default crop (relative to the active area) only, or all of it for `OpcodeList3` (cropped after)
        let c = if list3.is_empty() { crop } else { whole };
        let (bw, bh) = (c.width / k, c.height / k);
        if bw == 0 || bh == 0 {
            return Ok(None);
        }
        // Opcode lists 1/2 need the normalised copy; plain files are normalised row by row.
        let norm: Option<Normalized> = if self.opcodes.list1.is_empty() && self.opcodes.list2.is_empty() { None } else { Some(self.normalized()?) };
        let cfa = cfa.shifted(a.x, a.y);
        let range = self.white_at(0) - self.black.mean();
        let scale = if range > 0.0 { 1.0 / range } else { 1.0 };
        let uniform_black =
            self.black.delta_h.is_empty() && self.black.delta_v.is_empty() && self.black.values.iter().all(|&v| v == self.black.values[0]);
        let black0 = self.black.at(0, 0, 0, 1);
        // colour of every sample of a block, per pattern phase of the block's corner
        let (pw, ph) = (cfa.width, cfa.height);
        let layouts: Vec<(Vec<u8>, [f32; 3])> = (0..ph)
            .flat_map(|j| (0..pw).map(move |i| (i, j)))
            .map(|(i, j)| {
                let (l, n) = block_layout(&cfa, i, j, k);
                (l, n.map(|n| 1.0 / n.max(1) as f32))
            })
            .collect();
        let wlen = bw * k;
        // per pattern phase, each colour's samples as offsets into a block row's sample rows (`rows` below, from the
        // block's corner), in the block's order
        let positions: Vec<[Vec<usize>; 3]> = layouts
            .iter()
            .map(|(layout, _)| {
                std::array::from_fn(|ch| {
                    (0..k * k).filter(|&t| layout.get(t).is_some_and(|&l| l as usize == ch)).map(|t| (t / k) * wlen + t % k).collect()
                })
            })
            .collect();
        let mut out = Rgb32f::new(bw, bh);
        let mut mask = vec![0u8; bw * bh];
        out.data.par_chunks_mut(bw).zip(mask.par_chunks_mut(bw)).enumerate().for_each_init(Vec::new, |rows, (by, (row, mrow))| {
            // the block row's k sample rows, normalised (one buffer per thread: every sample is written below)
            rows.resize(k * wlen, 0.0);
            for dy in 0..k {
                let y = c.y + by * k + dy;
                let dst = &mut rows[dy * wlen..(dy + 1) * wlen];
                match (&norm, &self.data) {
                    (Some(nm), _) => dst.copy_from_slice(&nm.data[y * nm.width + c.x..][..wlen]),
                    (None, RawData::U16(d)) if uniform_black => {
                        let src = &d[(a.y + y) * self.width + a.x + c.x..][..wlen];
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = (v as f32 - black0) * scale;
                        }
                    }
                    (None, data) => {
                        let base = (a.y + y) * self.width + a.x + c.x;
                        for (x, o) in dst.iter_mut().enumerate() {
                            *o = (data.get(base + x) - self.black.at(c.x + x, y, 0, 1)) * scale;
                        }
                    }
                }
            }
            let py = (c.y + by * k) % ph;
            for (bx, (px, m)) in row.iter_mut().zip(mrow.iter_mut()).enumerate() {
                let phase = py * pw + (c.x + bx * k) % pw;
                let (inv, sites) = (&layouts[phase].1, &positions[phase]);
                // each colour's samples summed in the block's own order (row by row), in registers
                let (mut sum, mut max) = ([0f32; 3], [f32::MIN; 3]);
                for (ch, offsets) in sites.iter().enumerate() {
                    let (mut s, mut hi) = (0f32, f32::MIN);
                    for &o in offsets {
                        let v = rows.get(bx * k + o).copied().unwrap_or(0.0);
                        s += v;
                        hi = hi.max(v);
                    }
                    sum[ch] = s;
                    max[ch] = hi;
                }
                for ch in 0..3 {
                    let clipped = max[ch] >= clip;
                    px[ch] = if clipped && keep_max { max[ch] } else { sum[ch] * inv[ch] };
                    *m |= u8::from(clipped) << ch;
                }
            }
        });
        if align
            && cfa.is_bayer()
            && let Some((layout, _)) = layouts.get((c.y % ph) * pw + c.x % pw)
        {
            let offsets = plane_offsets(layout, k);
            // process V9, across edges: red and blue follow green's shape ([`follow_green`]), on the blocks the alignment
            // flags as candidates
            let rule = Follow::wider(self.sensor_noise(), k as f32 / 2.0);
            let greens = layout.iter().filter(|&&l| l == 1).count() as f32;
            if let Some((means, flags)) = align_and_flag(&mut out, offsets, Some(&rule), greens) {
                let (aw, ah) = (a.width, a.height);
                // a normalised sample of the active area, mirror-reflected at its edges (which keeps the pattern)
                let sample = |x: isize, y: isize| -> f32 {
                    let (x, y) = (crate::demosaic::reflect(x, aw), crate::demosaic::reflect(y, ah));
                    match (&norm, &self.data) {
                        (Some(nm), _) => nm.data.get(y * nm.width + x).copied().unwrap_or(0.0),
                        (None, RawData::U16(d)) if uniform_black => {
                            d.get((a.y + y) * self.width + a.x + x).map_or(0.0, |&v| (f32::from(v) - black0) * scale)
                        }
                        (None, data) => {
                            let i = (a.y + y) * self.width + a.x + x;
                            if i < data.len() { (data.get(i) - self.black.at(x, y, 0, 1)) * scale } else { 0.0 }
                        }
                    }
                };
                // the same away from the active area's edges, for a plain file (no reflection, no per-sample black)
                let plain = match (&norm, &self.data) {
                    (None, RawData::U16(d)) if uniform_black => Some(d),
                    _ => None,
                };
                let (stride, origin) = (self.width, a.y * self.width + a.x);
                // green estimated at block (bx, by)'s samples of `colour`, averaged (`green_at` reads 2 px around a site)
                let (cx, cy) = (c.x, c.y);
                let sites: [Vec<usize>; 3] =
                    std::array::from_fn(|colour| (0..layout.len()).filter(|&t| layout.get(t).is_some_and(|&l| l as usize == colour)).collect());
                let green_sites = |bx: usize, by: usize, colour: usize| -> f32 {
                    let (x0, y0) = (cx + bx * k, cy + by * k);
                    let Some(sites) = sites.get(colour).filter(|s| !s.is_empty()) else { return 0.0 };
                    let inside = plain.is_some() && x0 >= 2 && y0 >= 2 && x0 + k + 2 <= aw && y0 + k + 2 <= ah;
                    let sum: f32 = if let (true, Some(d)) = (inside, plain) {
                        let at = |t: usize| origin + (y0 + t / k) * stride + x0 + t % k;
                        sites.iter().map(|&t| green_at_plain(d, at(t), stride, black0, scale)).sum()
                    } else {
                        sites.iter().map(|&t| green_at(&sample, (x0 + t % k) as isize, (y0 + t / k) as isize)).sum()
                    };
                    sum / sites.len() as f32
                };
                let per_block: [f32; 3] = std::array::from_fn(|c| layout.iter().filter(|&&l| l as usize == c).count() as f32);
                // (a fold over the bytes, which the compiler can vectorise)
                let any_clipped = mask.par_chunks(bw).any(|row| row.iter().fold(0u8, |a, &m| a | m) != 0);
                follow_green(&mut out, &means, &flags, offsets, &rule, per_block, (&mask, any_clipped), &green_sites);
            }
        }
        if list3.is_empty() {
            return Ok(Some((out, mask)));
        }
        crate::opcodes::apply_list3(&list3, &mut out);
        // the default crop at the binned scale
        let (x0, y0) = ((crop.x / k).min(bw - 1), (crop.y / k).min(bh - 1));
        let (cw, ch) = ((crop.width / k).clamp(1, bw - x0), (crop.height / k).clamp(1, bh - y0));
        if (x0, y0, cw, ch) == (0, 0, bw, bh) {
            return Ok(Some((out, mask)));
        }
        let mask = (y0..y0 + ch).flat_map(|y| mask[y * bw + x0..y * bw + x0 + cw].iter().copied()).collect();
        Ok(Some((out.into_crop(x0, y0, cw, ch), mask)))
    }
}

/// How far each colour's samples lie from the centre of a `k × k` block, in blocks, `(x, y)`.
fn plane_offsets(layout: &[u8], k: usize) -> [(f32, f32); 3] {
    let centre = (k as f32 - 1.0) / 2.0;
    std::array::from_fn(|c| {
        let (mut x, mut y, mut n) = (0f32, 0f32, 0f32);
        for (t, _) in layout.iter().enumerate().filter(|(_, l)| **l as usize == c) {
            x += (t % k) as f32;
            y += (t / k) as f32;
            n += 1.0;
        }
        if n > 0.0 { ((x / n - centre) / k as f32, (y / n - centre) / k as f32) } else { (0.0, 0.0) }
    })
}

/// Samples' worth of green estimated at one site of a colour ([`green_at`], for the ratio's noise): a weighted mean of
/// the two horizontal and the two vertical greens, about two samples' variance.
const SITE_SAMPLES: f32 = 2.0;
/// Added to [`green_at`]'s gradients (of the white), so that a flat area weighs both directions alike.
const GRADIENT_FLOOR: f32 = 1e-3;
/// Block rows [`follow_green`] works on at a time (it estimates green at each needed block once per band, and those of
/// the 2 rows either side once more).
const BAND: usize = 32;

/// Green estimated at the red or blue site `(x, y)` of a Bayer mosaic read through `at`: the horizontal and the vertical
/// estimate (the mean of the two greens on either side), each weighted by how smooth its direction is (green's step
/// across the site and the site colour's second difference), so that next to an edge it follows the side the site
/// lies on.
fn green_at(at: &impl Fn(isize, isize) -> f32, x: isize, y: isize) -> f32 {
    let (l, r, u, d) = (at(x - 1, y), at(x + 1, y), at(x, y - 1), at(x, y + 1));
    let own = 2.0 * at(x, y);
    let dh = (l - r).abs() + (own - at(x - 2, y) - at(x + 2, y)).abs();
    let dv = (u - d).abs() + (own - at(x, y - 2) - at(x, y + 2)).abs();
    // weights 1 / (dh + floor) and 1 / (dv + floor), with a single division
    let (a, b) = (dh + GRADIENT_FLOOR, dv + GRADIENT_FLOOR);
    let g = ((l + r) * 0.5 * b + (u + d) * 0.5 * a) / (a + b);
    if g.is_finite() { g } else { 0.0 }
}

/// [`green_at`] at sample `i` of a plain file's raw data `d` (rows `stride` apart, black `black`, `scale` to the
/// white), reading the samples straight from rows sliced once: the same arithmetic, without a closure or a reflection
/// per read. 0 when the site's 5 × 5 cross reaches past the data.
fn green_at_plain(d: &[u16], i: usize, stride: usize, black: f32, scale: f32) -> f32 {
    let v = |s: u16| (f32::from(s) - black) * scale;
    let at = |j: Option<usize>| j.and_then(|j| d.get(j)).copied().map(v);
    let (Some(row), Some(u2), Some(u1), Some(d1), Some(d2)) = (
        i.checked_sub(2).and_then(|j| d.get(j..j + 5)),
        at(i.checked_sub(2 * stride)),
        at(i.checked_sub(stride)),
        at(i.checked_add(stride)),
        at(i.checked_add(2 * stride)),
    ) else {
        return 0.0;
    };
    let [l2, l, own, r, r2] = [0, 1, 2, 3, 4].map(|t| row.get(t).copied().map_or(0.0, v));
    let (u, dn) = (u1, d1);
    let own = 2.0 * own;
    let dh = (l - r).abs() + (own - l2 - r2).abs();
    let dv = (u - dn).abs() + (own - u2 - d2).abs();
    // weights 1 / (dh + floor) and 1 / (dv + floor), with a single division
    let (a, b) = (dh + GRADIENT_FLOOR, dv + GRADIENT_FLOOR);
    let g = ((l + r) * 0.5 * b + (u + dn) * 0.5 * a) / (a + b);
    if g.is_finite() { g } else { 0.0 }
}

/// The blocks a colour plane is read from along one axis, at block `i` of `n`, whose samples of the colour lie `d`
/// blocks from its centre: `[(block, weight); 4]`, the alignment's two ([`align_planes`]: the block and the one on the
/// other side of its centre, weights `1 - |d|` and `|d|`) and the next one out on either side ([`SUPPORT_OUTER`] in
/// all, split so that a linear ramp is still read at the block's centre). Blocks past the image's edge read its last.
fn axis(i: usize, n: usize, d: f32) -> [(usize, f32); 4] {
    let a = d.abs();
    let last = n.saturating_sub(1) as isize;
    let at = |k: isize| (i as isize + k).clamp(0, last) as usize;
    // the neighbour the alignment reads lies on the other side of the centre from the samples
    let s: isize = if d > 0.0 { -1 } else { 1 };
    [(at(0), 1.0 - a), (at(s), a), (at(-s), SUPPORT_OUTER * (2.0 - a) / 3.0), (at(2 * s), SUPPORT_OUTER * (1.0 + a) / 3.0)]
}

/// Process V9, every block: red and blue take green's shape across edges ([`crate::demosaic::edge`]).
/// [`align_planes`] reads each of them between its own block's samples and the next block's, which on a steep edge
/// takes red from more of one side than green and blue from more of the other: a violet line where a dark bridge
/// crosses a bright sky at fit size. Where the block's green (`means`, before alignment) and the green of a block the
/// alignment reads for a colour lie an edge's step apart ([`Follow::edge`]), the colour moves toward the block's green
/// times its ratio to green at its sites (`green_sites(x, y, c)`: green estimated at block `(x, y)`'s sites of colour
/// `c`), both read at the block's centre with [`align_planes`]' weights and summed over the alignment's blocks and the
/// next ones out ([`axis`]), kept within the alignment's blocks' means and trusted by the green's noise
/// ([`Follow::trust`]; `per_block`: the sites of each colour in a block). Only the blocks and colours `flags` marks
/// ([`align_and_flag`]) are looked at; elsewhere (flat and smoothly shaded areas) the aligned means stay, bit for
/// bit. A ratio, not a difference: across an edge the colour's ratio to green stays put, its difference doesn't.
///
/// The outer blocks next to a block holding a clipped sample (`mask`, `any_clipped`: whether any does) stay out of the
/// sums: their green at the sites reads clipped greens, which read lower than the light was, and a ratio to it
/// overstates the colour (a magenta line on the bright side of dark bars against a clipped sky, at -3 EV). The
/// alignment's own blocks stay in: they hold the colour being corrected, and the result is kept within their means.
///
/// In bands of [`BAND`] block rows, in parallel, each in three steps: the gate of every flagged colour; green at the
/// sites of every block a passing one reads, once each (kept per thread for the band and the rows around it); then
/// the ratio itself.
#[allow(clippy::too_many_arguments)]
fn follow_green(
    img: &mut Rgb32f,
    means: &Rgb32f,
    flags: &[u8],
    offsets: [(f32, f32); 3],
    rule: &Follow,
    per_block: [f32; 3],
    (mask, any_clipped): (&[u8], bool),
    green_sites: &(impl Fn(usize, usize, usize) -> f32 + Sync),
) {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 || img.data.len() != w * h || means.data.len() != w * h || flags.len() != w * h {
        return;
    }
    // samples in a block's green, for the edge gate's noise allowance
    let per_green = per_block.get(1).copied().unwrap_or(1.0);
    let at = |x: usize, y: usize| means.data.get(y * w + x).copied().unwrap_or([0.0; 3]);
    // the colours to move: (slot in the planes below, colour, offset of its samples)
    let colours: Vec<(usize, usize, f32, f32)> = [(0usize, 0usize), (1, 2)]
        .into_iter()
        .filter_map(|(slot, c)| offsets.get(c).filter(|(dx, dy)| *dx != 0.0 || *dy != 0.0).map(|&(dx, dy)| (slot, c, dx, dy)))
        .collect();
    // the alignment's four blocks for a colour ([`axis`]' first two along each)
    let four = |x: usize, y: usize, dx: f32, dy: f32| {
        let nx = if dx > 0.0 { x.saturating_sub(1) } else { (x + 1).min(w - 1) };
        let ny = if dy > 0.0 { y.saturating_sub(1) } else { (y + 1).min(h - 1) };
        [at(x, y), at(nx, y), at(x, ny), at(nx, ny)]
    };
    // the blocks a passing one reads along an axis ([`axis`]): from 1 before to 2 after it where the colour's samples
    // lie before the centre, from 2 before to 1 after otherwise
    let reach = |d: f32| if d > 0.0 { (2usize, 1usize) } else { (1usize, 2usize) };
    // which blocks hold a clipped sample or lie beside one (none in a frame without)
    let near_clip = if any_clipped { clip_touched(mask, w, h) } else { None };
    let touched = |x: usize, y: usize| near_clip.as_ref().is_some_and(|t| t.get(y * w + x).copied().unwrap_or(true));

    // per thread: the band's passing blocks (row, block, colour's index, gate), which guides each block needs, and the
    // guides of the band's rows and 2 either side
    type Scratch = (Vec<(usize, usize, usize, f32)>, Vec<u8>, Vec<[f32; 2]>);
    img.data.par_chunks_mut(w * BAND).zip(flags.par_chunks(w * BAND)).enumerate().for_each_init(
        || -> Scratch { (Vec::new(), Vec::new(), Vec::new()) },
        |(passing, marks, guides), (band, (rows, band_flags))| {
            let y0 = band * BAND;
            // 1. the edge gate on the greens of the alignment's four blocks
            passing.clear();
            for (r, flag_row) in band_flags.chunks(w).enumerate() {
                let y = y0 + r;
                for (x, &flag) in flag_row.iter().enumerate().filter(|(_, f)| **f != 0) {
                    let green = at(x, y)[1];
                    for (i, &(_, _, dx, dy)) in colours.iter().enumerate().filter(|(_, (_, c, _, _))| flag & (1 << *c) != 0) {
                        let (green_lo, green_hi) =
                            four(x, y, dx, dy).iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
                        let gate = rule.edge(green, green_lo, green_hi, per_green);
                        if gate.is_nan() || gate <= 0.0 {
                            continue;
                        }
                        passing.push((y, x, i, gate));
                    }
                }
            }
            if passing.is_empty() {
                return;
            }
            // 2. green at the sites of every block a passing block reads, each estimated once (rows `g0..` of the cache)
            let g0 = y0.saturating_sub(2);
            let span = (y0 + BAND + 2).min(h) - g0;
            marks.clear();
            marks.resize(span * w, 0);
            for &(y, x, i, _) in passing.iter() {
                let Some(&(slot, _, dx, dy)) = colours.get(i) else { continue };
                let ((xb, xa), (yb, ya)) = (reach(dx), reach(dy));
                for yy in y.saturating_sub(yb)..(y + ya + 1).min(h) {
                    let Some(row) = yy.checked_sub(g0).and_then(|r| marks.get_mut(r * w..(r + 1) * w)) else { continue };
                    for m in row.get_mut(x.saturating_sub(xb)..(x + xa + 1).min(w)).unwrap_or(&mut []) {
                        *m |= 1 << slot;
                    }
                }
            }
            // (only the marked entries are written and read: the others may hold an earlier band's)
            guides.resize(span * w, [0.0; 2]);
            for (j, (guide, &m)) in guides.iter_mut().zip(marks.iter()).enumerate().filter(|(_, (_, m))| **m != 0) {
                let (x, y) = (j % w, g0 + j / w);
                for &(slot, c, _, _) in &colours {
                    if m & (1 << slot) != 0
                        && let Some(g) = guide.get_mut(slot)
                    {
                        *g = green_sites(x, y, c);
                    }
                }
            }
            // 3. the ratio at every passing block and colour
            for &(y, x, i, gate) in passing.iter() {
                let (Some(px), Some(&(slot, c, dx, dy))) = (y.checked_sub(y0).and_then(|r| rows.get_mut(r * w + x)), colours.get(i)) else {
                    continue;
                };
                let green = at(x, y)[1];
                // the colour's range in the alignment's four blocks
                let (lo, hi) = four(x, y, dx, dy).iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p[c]), hi.max(p[c])));
                // and the blocks one further out, but those next to a clipped sample
                let (xs, ys) = (axis(x, w, dx), axis(y, h, dy));
                let (mut own_all, mut green_all, mut sum, mut sum_sq) = (0f32, 0f32, 0f32, 0f32);
                for (iy, &(ty, ky)) in ys.iter().enumerate() {
                    let means_row = means.data.get(ty * w..(ty + 1) * w);
                    let guide_row = ty.checked_sub(g0).and_then(|r| guides.get(r * w..(r + 1) * w));
                    let (Some(means_row), Some(guide_row)) = (means_row, guide_row) else { continue };
                    for (ix, &(tx, kx)) in xs.iter().enumerate() {
                        if (ix >= 2 || iy >= 2) && touched(tx, ty) {
                            continue;
                        }
                        let (Some(p), Some(g)) = (means_row.get(tx), guide_row.get(tx).and_then(|g| g.get(slot))) else { continue };
                        let k = kx * ky;
                        own_all += k * p[c];
                        green_all += k * g;
                        sum += k;
                        sum_sq += k * k;
                    }
                }
                if sum.is_nan() || sum <= 0.0 {
                    continue;
                }
                let (own_mean, guide_mean) = (own_all / sum, green_all / sum);
                let per = per_block.get(c).copied().unwrap_or(1.0) * SITE_SAMPLES;
                let t = gate * rule.trust(guide_mean, edge::samples(sum, sum_sq, per));
                let v = crate::demosaic::clamp_between(edge::ratio(green, own_mean, guide_mean), lo, hi);
                if t > 0.0 && v.is_finite() {
                    px[c] += (v - px[c]) * t;
                }
            }
        },
    );
}

/// Which blocks of a `w × h` clip `mask` hold a clipped sample or lie beside one (a 3 × 3 square), or `None` when none
/// holds one.
fn clip_touched(mask: &[u8], w: usize, h: usize) -> Option<Vec<bool>> {
    if w == 0 || mask.len() != w * h || mask.iter().all(|&m| m == 0) {
        return None;
    }
    // along the rows, then down the columns
    let mut rows = vec![false; w * h];
    rows.par_chunks_mut(w).zip(mask.par_chunks(w)).for_each(|(dst, src)| {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src.get(x.saturating_sub(1)..(x + 2).min(w)).is_some_and(|s| s.iter().any(|&m| m != 0));
        }
    });
    let mut out = vec![false; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        for yy in y.saturating_sub(1)..(y + 2).min(h) {
            if let Some(src) = rows.get(yy * w..(yy + 1) * w) {
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d |= s;
                }
            }
        }
    });
    Some(out)
}

/// Each colour plane read at the blocks' centres: between a block's own value (which lies `offsets` away from its
/// centre) and the neighbouring block's on the other side of it. Returns the block means as they were (`None`: nothing
/// to align).
#[cfg(test)]
fn align_planes(img: &mut Rgb32f, offsets: [(f32, f32); 3]) -> Option<Rgb32f> {
    align_and_flag(img, offsets, None, 1.0).map(|(means, _)| means)
}

/// [`align_planes`], and with a `rule`, per block which colours [`follow_green`] may change (bit `c` for colour `c`):
/// those the alignment moves, where the greens of the four blocks it reads for the colour (each the mean of
/// `per_green` samples) step far enough for the rule's edge gate ([`flag_row`], [`edge::FirstTest`]: a superset of
/// what the gate itself passes). Empty without a rule. In strips of rows, each keeping the greens of three rows
/// (gathered once per row) beside the rows it aligns.
fn align_and_flag(img: &mut Rgb32f, offsets: [(f32, f32); 3], rule: Option<&Follow>, per_green: f32) -> Option<(Rgb32f, Vec<u8>)> {
    /// Rows a strip holds.
    const STRIP: usize = 32;
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 || offsets.iter().all(|&(x, y)| x == 0.0 && y == 0.0) {
        return None;
    }
    // into a new picture, so that the means stay as they are without a copy (one written in parallel, as the copy
    // wasn't); each block starts as its means
    let mut aligned = Rgb32f::new(w, h);
    let src = &*img;
    let at = |x: usize, y: usize, c: usize| src.data.get(y * w + x).map_or(0.0, |p| p[c]);
    let align_row = |y: usize, row: &mut [[f32; 3]]| {
        for (x, px) in row.iter_mut().enumerate() {
            *px = src.data.get(y * w + x).copied().unwrap_or([0.0; 3]);
            for (c, &(dx, dy)) in offsets.iter().enumerate() {
                if dx == 0.0 && dy == 0.0 {
                    continue;
                }
                let nx = if dx > 0.0 { x.saturating_sub(1) } else { (x + 1).min(w - 1) };
                let ny = if dy > 0.0 { y.saturating_sub(1) } else { (y + 1).min(h - 1) };
                let (ax, ay) = (dx.abs(), dy.abs());
                px[c] =
                    at(x, y, c) * (1.0 - ax) * (1.0 - ay) + at(nx, y, c) * ax * (1.0 - ay) + at(x, ny, c) * (1.0 - ax) * ay + at(nx, ny, c) * ax * ay;
            }
        }
    };
    let Some(rule) = rule else {
        aligned.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| align_row(y, row));
        return Some((std::mem::replace(img, aligned), Vec::new()));
    };
    let gate = rule.first_test(per_green);
    // the colours to flag: (bit, the other row is above, the other column is to the left)
    let colours: Vec<(u8, bool, bool)> =
        offsets.iter().enumerate().filter(|(_, (dx, dy))| *dx != 0.0 || *dy != 0.0).map(|(c, &(dx, dy))| (1u8 << c, dy > 0.0, dx > 0.0)).collect();
    let greens = |y: usize, row: &mut Vec<f32>| {
        row.clear();
        row.extend(src.data.get(y * w..(y + 1) * w).unwrap_or(&[]).iter().map(|p| p[1]));
    };
    let mut flags = vec![0u8; w * h];
    aligned.data.par_chunks_mut(w * STRIP).zip(flags.par_chunks_mut(w * STRIP)).enumerate().for_each(|(strip, (rows, flag_rows))| {
        let y0 = strip * STRIP;
        let mut buf: [Vec<f32>; 5] = Default::default();
        let [up, mid, down, lo, hi] = &mut buf;
        greens(y0.saturating_sub(1), up);
        greens(y0, mid);
        for (r, (row, flag_row)) in rows.chunks_mut(w).zip(flag_rows.chunks_mut(w)).enumerate() {
            let y = y0 + r;
            align_row(y, row);
            greens((y + 1).min(h - 1), down);
            for &colour in &colours {
                flag_row_of(&gate, colour, (up, mid, down), (lo, hi), flag_row);
            }
            // the window moves a row down
            std::mem::swap(up, mid);
            std::mem::swap(mid, down);
        }
    });
    Some((std::mem::replace(img, aligned), flags))
}

/// Flag `bit` in `flag_row` (bit, `above`: the colour's other row is above, `left`: its other column is to the left)
/// where the greens of the four blocks the alignment reads for the colour step far enough for `gate`: their extremes
/// down the two rows (`up`, `mid`, `down`: the greens of the row above, the row and the row below), then across the two
/// columns (`lo`, `hi`: room), in straight loops over slices the compiler can vectorise (plain comparisons: a NaN green
/// only flags or doesn't).
fn flag_row_of(
    gate: &edge::FirstTest,
    (bit, above, left): (u8, bool, bool),
    (up, mid, down): (&[f32], &[f32], &[f32]),
    (lo, hi): (&mut Vec<f32>, &mut Vec<f32>),
    flag_row: &mut [u8],
) {
    let low = |a: f32, b: f32| if b < a { b } else { a };
    let high = |a: f32, b: f32| if b > a { b } else { a };
    let n = mid.len().min(flag_row.len());
    // down the two rows
    let other = if above { up } else { down };
    lo.clear();
    lo.extend(mid.iter().zip(other).map(|(&m, &o)| low(m, o)));
    hi.clear();
    hi.extend(mid.iter().zip(other).map(|(&m, &o)| high(m, o)));
    // and across the two columns: block x and x - 1 (left) or x + 1, the edge block again at the edge
    let at = |v: &[f32], i: usize| v.get(i).copied().unwrap_or(0.0);
    let test = |g: f32, l: f32, h: f32| bit * u8::from(gate.passes(g, l, h));
    let edge_x = if left { 0 } else { n.saturating_sub(1) };
    if let Some(flag) = flag_row.get_mut(edge_x)
        && edge_x < n
    {
        *flag |= test(at(mid, edge_x), at(lo, edge_x), at(hi, edge_x));
    }
    if n >= 2 {
        let (lo, hi) = (lo.get(..n).unwrap_or(&[]), hi.get(..n).unwrap_or(&[]));
        let pairs = lo.iter().zip(lo.iter().skip(1)).zip(hi.iter().zip(hi.iter().skip(1)));
        if left {
            for ((flag, &g), ((&l0, &l1), (&h0, &h1))) in flag_row.iter_mut().skip(1).zip(mid.iter().skip(1)).zip(pairs) {
                *flag |= test(g, low(l1, l0), high(h1, h0));
            }
        } else {
            for ((flag, &g), ((&l0, &l1), (&h0, &h1))) in flag_row.iter_mut().zip(mid.iter()).zip(pairs) {
                *flag |= test(g, low(l0, l1), high(h0, h1));
            }
        }
    }
}

/// Colours of the `k × k` samples of a block whose corner sits at pattern position `(i, j)`
/// (row-major) and the count of each colour.
fn block_layout(cfa: &crate::Cfa, i: usize, j: usize, k: usize) -> (Vec<u8>, [u32; 3]) {
    let mut n = [0u32; 3];
    let l: Vec<u8> = (0..k * k)
        .map(|t| {
            let c = cfa.color_at(i + t % k, j + t / k).min(2);
            n[c as usize] += 1;
            c
        })
        .collect();
    (l, n)
}

#[cfg(test)]
mod tests {
    use crate::demosaic::mosaic_from_rgb;
    use crate::*;

    fn raw_from(n: &Normalized, black: f32, white: f32) -> RawImage {
        let data = n.data.iter().map(|&v| (v * (white - black) + black).round() as u16).collect();
        RawImage {
            format: RawFormat::Dng,
            width: n.width,
            height: n.height,
            cpp: 1,
            data: RawData::U16(data),
            cfa: n.cfa.clone(),
            bits: 14,
            black: BlackLevel::uniform(black),
            white: vec![white],
            active_area: Rect::new(0, 0, n.width, n.height),
            crop: Rect::new(0, 0, n.width, n.height),
            orientation: Orientation::Normal,
            color: ColorData::default(),
            wb_multipliers: None,
            linearized: false,
            pixel_aspect: 1.0,
            opcodes: OpcodeLists::default(),
            metadata: Metadata::default(),
            lens_metadata: Default::default(),
        }
    }

    /// A hard edge through the middle of the blocks: the plain block has its red from above the edge and its blue from
    /// below (the coloured line of a reduced-size render), the aligned one nearly the same in both.
    #[test]
    fn aligned_binning_keeps_a_hard_edge_free_of_colour() {
        let img = Rgb32f::from_fn(24, 24, |_, y| if y < 9 { [0.02; 3] } else { [0.8; 3] });
        let raw = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap()), 0.0, 16383.0);
        let plain = raw.develop_binned_masked(2, 0.99).unwrap().unwrap().0;
        let aligned = raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap().0;
        let spread = |i: &Rgb32f| (i.get(5, 4)[2] - i.get(5, 4)[0]).abs();
        assert!(spread(&plain) > 0.7, "{:?}", plain.get(5, 4));
        assert!(spread(&aligned) < 0.6 * spread(&plain), "{:?}", aligned.get(5, 4));
        // away from the edge nothing moves
        assert_eq!(aligned.get(5, 1), plain.get(5, 1));
        assert_eq!(aligned.get(5, 9), plain.get(5, 9));
    }

    #[test]
    fn binning_matches_block_means_and_keeps_clipping() {
        let img = Rgb32f::from_fn(24, 18, |x, y| [0.1 + x as f32 * 0.01, 0.2 + y as f32 * 0.01, 0.3]);
        let mut raw = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer("GRBG").unwrap()), 512.0, 16383.0);
        assert!(raw.can_bin(2) && raw.can_bin(4) && !raw.can_bin(3));
        let b = raw.develop_binned(2, 0.99).unwrap().unwrap();
        assert_eq!((b.width, b.height), (12, 9));
        let p = b.get(3, 2);
        // red sample of block (3,2) under GRBG is at (7,4); blue at (6,5); greens average (6,4)+(7,5)
        assert!((p[0] - img.get(7, 4)[0]).abs() < 1e-3, "{p:?}");
        assert!((p[2] - 0.3).abs() < 1e-3);
        assert!((p[1] - (img.get(6, 4)[1] + img.get(7, 5)[1]) / 2.0).abs() < 1e-3);
        // a single clipped green sample keeps the block's green clipped
        if let RawData::U16(d) = &mut raw.data {
            d[4 * 24 + 6] = 16383;
        }
        assert!(raw.develop_binned(2, 0.99).unwrap().unwrap().get(3, 2)[1] >= 0.99);
        // default crop is honoured at the binned scale
        raw.crop = Rect::new(2, 2, 20, 12);
        let c = raw.develop_binned(2, 0.99).unwrap().unwrap();
        assert_eq!((c.width, c.height), (10, 6));
        // X-Trans: 3×3 blocks hold all colours, 2×2 don't
        let x = raw_from(&mosaic_from_rgb(&img, &Cfa::xtrans()), 0.0, 1000.0);
        assert!(x.can_bin(3) && x.can_bin(6) && !x.can_bin(2));
        let xb = x.develop_binned(3, 0.99).unwrap().unwrap();
        assert_eq!((xb.width, xb.height), (8, 6));
        assert!((xb.get(4, 3)[2] - 0.3).abs() < 2e-3);
    }

    /// Issue #548: specular points that clip one green sample in some 8 × 8 blocks of a grey surface.
    /// Taking the block's maximum makes those blocks' green the clip level, which reconstruction then
    /// keeps (a green speckle); the masked binning keeps the mean and marks the green as clipped.
    #[test]
    fn clipping_is_decided_before_averaging() {
        let img = Rgb32f::filled(64, 64, [0.3, 0.3, 0.3]);
        let mut raw = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap()), 0.0, 4095.0);
        if let RawData::U16(d) = &mut raw.data {
            // one green sample (RGGB: odd column of an even row) in every other block
            for by in 0..8 {
                for bx in (by % 2..8).step_by(2) {
                    d[(by * 8 + 2) * 64 + bx * 8 + 3] = 4095;
                }
            }
        }
        let wb = [1.0f32; 3];
        let (mut masked, mask) = raw.develop_binned_masked(8, 0.99).unwrap().unwrap();
        assert_eq!(mask.len(), 64);
        assert_eq!(mask.iter().filter(|&&m| m == highlight::CLIPPED_G).count(), 32);
        assert!(mask.iter().all(|&m| m & (highlight::CLIPPED_R | highlight::CLIPPED_B) == 0));
        // the clipped block's green is its mean: (31 · 0.3 + 1) / 32
        assert!((masked.get(0, 0)[1] - (31.0 * 0.3 + 1.0) / 32.0).abs() < 1e-3, "{:?}", masked.get(0, 0));
        assert_eq!(highlight::reconstruct_masked(&mut masked, wb, 0.99, [1.0; 3], &mask), 32);
        let green = |img: &Rgb32f| img.data.iter().map(|p| p[1] / p[0]).fold(0.0f32, f32::max);
        // (no more than the mean itself: the clipped sample counts at the clip level)
        assert!(green(&masked) < 1.08, "green excess {}", green(&masked));
        // the maximum rule: a green speckle at the clip level
        let mut max = raw.develop_binned(8, 0.99).unwrap().unwrap();
        assert!(max.get(0, 0)[1] >= 0.99);
        highlight::reconstruct(&mut max, wb, 0.99);
        assert!(green(&max) > 3.0, "{}", green(&max));
        // without clipped samples both are the same plain means, and the mask is empty
        let clean = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap()), 0.0, 4095.0);
        let (a, m) = clean.develop_binned_masked(8, 0.99).unwrap().unwrap();
        assert!(m.iter().all(|&v| v == 0));
        assert!(a.data == clean.develop_binned(8, 0.99).unwrap().unwrap().data);
    }

    #[test]
    fn binning_matches_demosaic_downscaled_on_smooth_images() {
        let img = Rgb32f::from_fn(64, 48, |x, y| [0.2 + 0.3 * (x as f32 / 64.0), 0.4 - 0.2 * (y as f32 / 48.0), 0.25]);
        let raw = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap()), 0.0, 4095.0);
        let b = raw.develop_binned(2, 0.99).unwrap().unwrap();
        let full = lcc_raster::resample::half(&raw.develop(Method::Ahd).unwrap());
        for y in 2..b.height - 2 {
            for x in 2..b.width - 2 {
                for c in 0..3 {
                    assert!((b.get(x, y)[c] - full.get(x, y)[c]).abs() < 0.01, "({x},{y})");
                }
            }
        }
    }

    /// [`RawImage::develop_binned_masked_aligned`] as it was before red and blue followed green: the block means, their
    /// colour planes aligned (default crop and active area at the origin).
    fn plain_aligned(raw: &RawImage, k: usize) -> (Rgb32f, Vec<u8>) {
        let (mut img, mask) = raw.develop_binned_masked(k, 0.99).unwrap().unwrap();
        let (layout, _) = super::block_layout(raw.cfa.as_ref().unwrap(), 0, 0, k);
        super::align_planes(&mut img, super::plane_offsets(&layout, k));
        (img, mask)
    }

    const PATTERNS: [&str; 4] = ["RGGB", "BGGR", "GRBG", "GBRG"];

    /// A dark beam across a pale blue sky that brightens away from it until green, then blue clip a few pixels from its
    /// edges (red never does), as at the edge of a clipped sky. The edges are soft, as a lens draws them, and slanted,
    /// so they cross the blocks at every phase, sky above one and below the other.
    fn beam(peak: f32) -> Rgb32f {
        let (sky, dark) = ([0.38f32, 1.0, 0.82], [0.02f32, 0.045, 0.03]);
        Rgb32f::from_fn(64, 64, |x, y| {
            // distance out of the 10 px beam (negative inside)
            let d = (y as f32 + 0.5 - (30.0 + x as f32 / 8.0)).abs() - 5.0;
            let t = 0.5 + 0.5 * (d / 0.8).tanh();
            let level = peak * (0.92 + 0.08 * d.clamp(0.0, 6.0));
            std::array::from_fn(|c| (dark[c] + (sky[c] * level - dark[c]) * t).min(1.0))
        })
    }

    /// A dark bridge's girders against a bright sky that is nowhere near its clip (the bridge of test frame 6719):
    /// soft edges as a lens draws them, one slanted and one steep, so they cross the blocks at every phase. Camera RGB:
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

    /// Blocks on an edge (green between 15 % and 85 % of the brightest green within `reach` blocks) whose red or blue
    /// ratio to green lies more than 8 % outside the range the two sides span (`rg`, `bg`: colour neither side has, the
    /// line of a fit-size render), and the count of edge blocks.
    fn false_colour(img: &Rgb32f, reach: usize, rg: (f32, f32), bg: (f32, f32)) -> (usize, usize) {
        let (mut bad, mut n) = (0, 0);
        let (w, h) = (img.width, img.height);
        for y in 0..h {
            for x in 0..w {
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
                if !(rg.0 / 1.08..=rg.1 * 1.08).contains(&r) || !(bg.0 / 1.08..=bg.1 * 1.08).contains(&b) {
                    bad += 1;
                }
            }
        }
        (bad, n)
    }

    /// Next to a clipped sky an edge's blocks (which hold no clipped sample themselves) took red and blue from either
    /// side of it (a lavender line at fit size); following green's shape they keep the colours of the two sides. Deep
    /// inside the clipped sky the blocks bin exactly as before.
    #[test]
    fn next_to_a_clipped_sky_an_edge_keeps_its_colours() {
        let (mut before, mut total) = (0, 0);
        for pattern in PATTERNS {
            let raw = raw_from(&mosaic_from_rgb(&beam(1.0), &Cfa::bayer(pattern).unwrap()), 0.0, 16383.0);
            let (plain, mask) = plain_aligned(&raw, 2);
            let (fixed, mask_fixed) = raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap();
            assert_eq!(mask, mask_fixed, "{pattern}: the clip mask is the same");
            assert!(mask.iter().filter(|&&m| m != 0).count() > 100, "{pattern}: the sky clips");
            // red / green: sky 0.38 (unclipped) to 0.53 (green clipped), ceiling 0.44; blue / green: 0.67 to 1.0
            let measure = |img: &Rgb32f| false_colour(img, 4, (0.38, 0.53), (0.67, 1.0));
            let (was, n) = measure(&plain);
            let (after, _) = measure(&fixed);
            eprintln!("{pattern}: false colour on the edges {was} -> {after} of {n}");
            assert!(n >= 60, "{pattern}: {n} edge blocks");
            assert!(after * 30 <= n, "{pattern}: {after} of {n}");
            (before, total) = (before + was, total + n);
            // deep inside the clipped sky (every block within 3 clipped): bit for bit
            let reach = 3;
            let (w, h) = (plain.width, plain.height);
            let mut deep = 0;
            for y in reach..h - reach {
                for x in reach..w - reach {
                    if (y - reach..=y + reach).all(|yy| (x - reach..=x + reach).all(|xx| mask[yy * w + xx] != 0)) {
                        deep += 1;
                        assert_eq!(fixed.get(x, y), plain.get(x, y), "{pattern} ({x}, {y})");
                    }
                }
            }
            assert!(deep > 50, "{pattern}: {deep} blocks deep in the clipped sky");
        }
        assert!(before * 10 >= total, "plain aligned binning draws the line: {before} of {total}");
    }

    /// On a bright edge that clips nowhere, the aligned planes took red from more of one side than green and blue from
    /// more of the other (a violet line at fit size); following green's shape the edge keeps the sides' colours.
    /// Without the edge rule (process V9 before this change) the line is there.
    #[test]
    fn a_bright_unclipped_edge_keeps_its_colours() {
        let (mut before, mut after, mut total) = (0, 0, 0);
        for pattern in PATTERNS {
            let mut raw = raw_from(&mosaic_from_rgb(&bridge(128, 96), &Cfa::bayer(pattern).unwrap()), 512.0, 16383.0);
            raw.metadata.iso = Some(100);
            let (plain, mask) = plain_aligned(&raw, 2);
            assert!(mask.iter().all(|&m| m == 0), "{pattern}: nothing clips");
            let fixed = raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap().0;
            let measure = |img: &Rgb32f| false_colour(img, 3, (0.45, 0.5), (0.75, 0.8));
            let (was, n) = measure(&plain);
            let (now, _) = measure(&fixed);
            eprintln!("{pattern}: line on the edges {was} -> {now} of {n}");
            assert!(n >= 60, "{pattern}: {n} edge blocks");
            (before, after, total) = (before + was, after + now, total + n);
        }
        assert!(before * 5 >= total, "without the rule the edge has a line: {before} of {total}");
        assert!(after * 30 <= total, "with it: {after} of {total}");
    }

    /// Dark bars against a sky two stops over the clip in green (blue clips too, red doesn't): the ring next to clipped
    /// samples, where the bars' edges climb toward the clipped sky, that a -3 EV render shows. Soft edges, one bar
    /// slanted and one steep, every block phase. Camera RGB: sky red / green 0.45, blue / green 0.75.
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

    /// Over the blocks of `out` (×`k`) within 5 sensor pixels of a block holding a clipped sample (`mask`) that hold
    /// none themselves and whose true colour (`truth`, the scene's block means) is clipped in no channel: how many have
    /// a red or blue ratio to green more than 10 % outside the range the bars and the sky span (colour neither has),
    /// how many of those are magenta (red and blue both too high against green), and the count.
    fn ring_error(out: &Rgb32f, truth: &Rgb32f, mask: &[u8], k: usize) -> (usize, usize, usize) {
        let (w, h) = (out.width, out.height);
        let r = 5usize.div_ceil(k);
        let (mut bad, mut magenta, mut count) = (0, 0, 0);
        for y in 0..h {
            for x in 0..w {
                let t: [f32; 3] =
                    std::array::from_fn(|c| (0..k * k).map(|i| truth.get(x * k + i % k, y * k + i / k)[c]).sum::<f32>() / (k * k) as f32);
                let near =
                    (y.saturating_sub(r)..(y + r + 1).min(h)).any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| mask[yy * w + xx] != 0));
                // (a block holding a clipped sample is the highlight rebuild's)
                if t.iter().any(|&v| v >= 0.95) || mask[y * w + x] != 0 || !near {
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

    /// Process V9 next to clipped samples, at fit size (×2 and ×4): where a dark bar's edge climbs toward a clipped sky
    /// (the ring a -3 EV render shows), red and blue don't turn magenta. A ratio to a green read from clipped greens,
    /// which read too low, made them too high there; the aligned means alone are magenta here too (they read the
    /// clipped sky's planes), and the rule takes most of it away.
    #[test]
    fn next_to_clipped_samples_the_ring_keeps_its_colour() {
        let truth = ring_scene(224, 160);
        let clipped = Rgb32f { data: truth.data.iter().map(|p| p.map(|v| v.min(1.0))).collect(), ..truth.clone() };
        let mut results = Vec::new();
        for k in [2, 4] {
            let (mut plain_bad, mut plain_magenta, mut bad, mut magenta, mut total) = (0, 0, 0, 0, 0);
            for pattern in PATTERNS {
                let mut raw = raw_from(&mosaic_from_rgb(&clipped, &Cfa::bayer(pattern).unwrap()), 512.0, 16383.0);
                raw.metadata.iso = Some(100);
                let (plain, mask) = plain_aligned(&raw, k);
                let fixed = raw.develop_binned_masked_aligned(k, 0.99).unwrap().unwrap().0;
                let (was, was_magenta, count) = ring_error(&plain, &truth, &mask, k);
                let (now, now_magenta, _) = ring_error(&fixed, &truth, &mask, k);
                eprintln!("×{k} {pattern}: off colour in the ring {was} ({was_magenta} magenta) -> {now} ({now_magenta} magenta) of {count}");
                (plain_bad, plain_magenta, bad, magenta, total) =
                    (plain_bad + was, plain_magenta + was_magenta, bad + now, magenta + now_magenta, total + count);
            }
            eprintln!("×{k} ring: {plain_bad} ({plain_magenta} magenta) -> {bad} ({magenta} magenta) of {total}");
            results.push((k, plain_magenta, magenta, total));
        }
        for (k, plain_magenta, magenta, total) in results {
            assert!(total >= 100, "×{k}: {total} ring blocks");
            // the aligned means alone are magenta here (they read the clipped sky's planes); the rule takes most of it
            assert!(magenta * 2 <= plain_magenta, "×{k}: magenta in the ring: {plain_magenta} -> {magenta} of {total}");
            if k == 2 {
                assert!(magenta * 10 <= total, "×{k}: magenta in the ring: {magenta} of {total}");
            }
        }
    }

    /// Flat and smoothly shaded areas (a flat colour, linear ramps, a slow colour wave, a bright gradient with mild noise)
    /// bin bit for bit as the aligned means, at ×2 and ×4, high ISO or low.
    #[test]
    fn flat_and_smooth_areas_bin_as_before() {
        let (w, h) = (96, 64);
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
                let mut raw = raw_from(&mosaic_from_rgb(img, &Cfa::bayer(pattern).unwrap()), 512.0, 16383.0);
                for iso in [Some(100), Some(3200), None] {
                    raw.metadata.iso = iso;
                    for k in [2, 4] {
                        let fixed = raw.develop_binned_masked_aligned(k, 0.99).unwrap().unwrap().0;
                        let was = plain_aligned(&raw, k).0;
                        let moved: Vec<(usize, usize)> =
                            (0..was.data.len()).filter(|&i| fixed.data[i] != was.data[i]).map(|i| (i % was.width, i / was.width)).collect();
                        assert!(moved.is_empty(), "{name} {pattern} {iso:?} ×{k}: {} blocks moved, {:?}", moved.len(), &moved[..moved.len().min(12)]);
                    }
                }
            }
        }
    }

    /// A dark, noisy scene shot at a high ISO (girders and textured shadows at 1-8 % of the white): the edge rule moves
    /// no channel's mean against green's. A ratio to a green that noise dominates comes out too high on average (the
    /// magenta shadows of a first try, a* +2.3 at L* < 20); the noise model keeps the rule out of them.
    #[test]
    fn a_dark_noisy_frame_keeps_its_colour() {
        let (w, h) = (192, 128);
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
        for pattern in ["RGGB", "GBRG"] {
            let mut n = mosaic_from_rgb(&img, &Cfa::bayer(pattern).unwrap());
            let mut raw = raw_from(&n, 512.0, 16383.0);
            raw.metadata.iso = Some(6400);
            // Gaussian noise of the file's own model (sum of four uniforms)
            let noise = raw.sensor_noise();
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            for v in n.data.iter_mut() {
                let g = (0..4)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .sum::<f32>()
                    * 3f32.sqrt();
                *v = (*v + g * noise.variance(*v).sqrt()).max(0.0);
            }
            raw = RawImage { metadata: raw.metadata.clone(), ..raw_from(&n, 512.0, 16383.0) };
            let was = means(&plain_aligned(&raw, 2).0);
            let now = means(&raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap().0);
            // the same mosaic told it came from ISO 100: the ratio runs in the noise
            raw.metadata.iso = Some(100);
            let wrong = means(&raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap().0);
            for c in 0..2 {
                let (shift, wrong) = (now[c] / was[c] - 1.0, wrong[c] / was[c] - 1.0);
                eprintln!("{pattern}: {} / green {shift:+.5} (as if ISO 100: {wrong:+.5})", ["red", "blue"][c]);
                assert!(shift.abs() < 0.002, "{pattern} channel {c}: {shift}");
            }
        }
    }

    /// Small clipped glints on a dark, noisy surface (catchlights in an eye): the dark blocks around them keep their
    /// colour on average (red and blue as a ratio to a green that noise dominates would come out too high: magenta).
    #[test]
    fn dark_noise_around_glints_keeps_its_colour() {
        let img = Rgb32f::from_fn(96, 64, |x, y| if x % 16 == 7 && y % 16 == 7 { [1.0; 3] } else { [0.012, 0.025, 0.018] });
        let mut n = mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap());
        // deterministic noise, about 0.004 of white per sample (an ISO 800 file's at this level)
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for v in n.data.iter_mut().filter(|v| **v < 0.5) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v += ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.014;
        }
        let mut raw = raw_from(&n, 512.0, 16383.0);
        raw.metadata.iso = Some(800);
        let (plain, mask) = plain_aligned(&raw, 2);
        let (fixed, _) = raw.develop_binned_masked_aligned(2, 0.99).unwrap().unwrap();
        let (w, h) = (plain.width, plain.height);
        let (mut sum_plain, mut sum_fixed, mut count) = ([0f64; 3], [0f64; 3], 0);
        for y in 0..h {
            for x in 0..w {
                let near = (y.saturating_sub(3)..(y + 4).min(h)).any(|yy| (x.saturating_sub(3)..(x + 4).min(w)).any(|xx| mask[yy * w + xx] != 0));
                let at_glint = (y.saturating_sub(1)..(y + 2).min(h)).any(|yy| (x.saturating_sub(1)..(x + 2).min(w)).any(|xx| mask[yy * w + xx] != 0));
                if near && !at_glint {
                    count += 1;
                    for c in 0..3 {
                        sum_plain[c] += plain.get(x, y)[c] as f64;
                        sum_fixed[c] += fixed.get(x, y)[c] as f64;
                    }
                }
            }
        }
        assert!(count > 200, "{count} dark blocks near the glints");
        for c in [0, 2] {
            let ratio = sum_fixed[c] / sum_plain[c];
            eprintln!("channel {c}: mean {ratio:.4} of plain aligned");
            assert!((ratio - 1.0).abs() < 0.01, "channel {c}: {ratio}");
        }
    }

    /// Clipped samples in tiny or odd-sized frames, and a float raw with non-finite samples: binning never panics.
    #[test]
    fn clipped_samples_in_odd_frames_never_panic() {
        for (w, h) in [(2, 2), (4, 2), (2, 4), (6, 6), (10, 4), (4, 10)] {
            for pattern in ["RGGB", "GBRG"] {
                let img = Rgb32f::from_fn(w, h, |x, y| if (x + 2 * y) % 3 == 0 { [1.0; 3] } else { [0.05, 0.1, 0.07] });
                let raw = raw_from(&mosaic_from_rgb(&img, &Cfa::bayer(pattern).unwrap()), 0.0, 4095.0);
                for k in [2, 4] {
                    if let Some((b, _)) = raw.develop_binned_masked_aligned(k, 0.99).unwrap() {
                        assert!(b.data.iter().all(|p| p.iter().all(|v| v.is_finite())), "{w}×{h} {pattern} ×{k}");
                    }
                }
            }
        }
        for img in [beam(1.0), bridge(64, 48)] {
            let n = mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap());
            let mut raw = raw_from(&n, 0.0, 16383.0);
            let mut data = n.data.clone();
            for (i, v) in data.iter_mut().enumerate().filter(|(i, _)| i % 97 == 0) {
                *v = if i % 2 == 0 { f32::NAN } else { f32::INFINITY };
            }
            raw.data = RawData::F32(data);
            raw.white = vec![1.0];
            assert!(raw.develop_binned_masked_aligned(2, 0.99).unwrap().is_some());
        }
    }
}

#[cfg(test)]
mod list3_tests {
    use crate::demosaic::mosaic_from_rgb;
    use crate::opcodes::{Area, Opcode};
    use crate::*;

    fn raw(list3: Vec<Opcode>) -> RawImage {
        let img = Rgb32f::from_fn(32, 24, |x, y| [0.1 + x as f32 * 0.005, 0.2 + y as f32 * 0.005, 0.25]);
        let n = mosaic_from_rgb(&img, &Cfa::bayer("RGGB").unwrap());
        RawImage {
            format: RawFormat::Dng,
            width: n.width,
            height: n.height,
            cpp: 1,
            data: RawData::U16(n.data.iter().map(|&v| (v * 16383.0).round() as u16).collect()),
            cfa: n.cfa.clone(),
            bits: 14,
            black: BlackLevel::uniform(0.0),
            white: vec![16383.0],
            active_area: Rect::new(0, 0, n.width, n.height),
            crop: Rect::new(2, 2, 28, 20),
            orientation: Orientation::Normal,
            color: ColorData::default(),
            wb_multipliers: None,
            linearized: false,
            pixel_aspect: 1.0,
            opcodes: OpcodeLists { list3, ..Default::default() },
            metadata: Metadata::default(),
            lens_metadata: Default::default(),
        }
    }

    fn area() -> Area {
        Area { top: 0, left: 0, bottom: 24, right: 32, plane: 0, planes: 3, row_pitch: 1, col_pitch: 1 }
    }

    /// A lens-shading gain map in `OpcodeList3` (as DJI DNGs carry) no longer forces a full-size develop: the binned
    /// image gets it at its own scale, and the default crop after it.
    #[test]
    fn a_gain_map_applies_to_the_binned_image() {
        let gain =
            Opcode::GainMap { area: area(), points_v: 2, points_h: 2, spacing: [1.0, 1.0], origin: [0.0, 0.0], map_planes: 1, gains: vec![2.0; 4] };
        let plain = raw(Vec::new()).develop_binned(2, 0.99).unwrap().unwrap();
        let shaded = raw(vec![gain]).develop_binned(2, 0.99).unwrap().expect("binned despite OpcodeList3");
        assert_eq!((shaded.width, shaded.height), (plain.width, plain.height));
        assert_eq!((plain.width, plain.height), (14, 10), "the default crop at the binned scale");
        for (a, b) in plain.data.iter().zip(&shaded.data) {
            for c in 0..3 {
                assert!((b[c] - 2.0 * a[c]).abs() < 1e-4, "{a:?} {b:?}");
            }
        }
    }

    /// An operation that needs every pixel (per-row deltas) still refuses binning: the caller demosaics.
    #[test]
    fn a_per_row_opcode_refuses_binning() {
        let rows = Opcode::DeltaPerRow { area: area(), deltas: vec![0.01; 24] };
        assert!(raw(vec![rows]).develop_binned(2, 0.99).unwrap().is_none());
        assert!(
            Opcode::GainMap {
                area: Area { row_pitch: 2, ..area() },
                points_v: 1,
                points_h: 1,
                spacing: [1.0, 1.0],
                origin: [0.0, 0.0],
                map_planes: 1,
                gains: vec![1.0]
            }
            .binned(2)
            .is_none()
        );
    }
}
