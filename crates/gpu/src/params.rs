//! Parameter blocks: the per-pixel stage's [`FinishParams`] packed into the 32-bit words the
//! `finish` kernel reads (`F_*` constants, generated from [`FIELDS`]), plus its auxiliary tables.

use lcc_color::spline::Lut1;
use lcc_develop::VignetteStyle;
use lcc_pipeline::colorops::{HSL_WORDS, MIXER_WORDS, Mixer, POINT_WORDS};
use lcc_pipeline::finish::{CURVE_LUT_N, FinishParams, MASK_TERMS, srgb_lut};

use crate::ctx::{FailKind, fail};

/// Entries per tone-curve table (`lcc_pipeline::finish` builds them at this size).
pub const CURVE_N: u32 = CURVE_LUT_N as u32;

/// Words per mask in the local colour records: the number of Point Color samples, whether the mask
/// has a Color Mixer, the mixer's [`MIXER_WORDS`], then up to 8 samples of [`POINT_WORDS`].
pub const LOCAL_POINT_WORDS: usize = 2 + MIXER_WORDS + 8 * POINT_WORDS;
/// Offset of the first Point Color sample in a local colour record.
pub const LOCAL_POINT_SAMPLES: usize = 2 + MIXER_WORDS;

/// Field name and number of 32-bit words.
const FIELDS: &[(&str, usize)] = &[
    ("W", 1),
    ("H", 1),
    ("NMASK", 1),
    ("MASK_OFF", 1),
    ("SRGB_OFF", 1),
    ("CURVE_OFF", 1),
    ("CURVES", 1),
    // process V9's Tone Curve before the output transform (`FinishParams::curves_pp`): four tables at PPCURVE_OFF
    // (parametric ∘ master, red, green, blue) and the working ↔ ProPhoto matrices
    ("PPCURVES", 1),
    ("PPCURVE_OFF", 1),
    ("PP_M", 9),
    ("PP_INV", 9),
    // the masks' local Curves: a slot per mask (its table set, or −1), then the table sets
    ("LCURVE_OFF", 1),
    // the masks' local Point Color: one record of `LOCAL_POINT_WORDS` per mask in `aux`
    ("LPOINTS", 1),
    ("LPOINT_OFF", 1),
    ("LCURVES", 1),
    ("GAIN", 1),
    ("EV", 1),
    ("AIR", 1),
    ("AIR_PRE", 1),
    ("HL", 1),
    ("SH", 1),
    // process V9's global Highlights / Shadows (`FinishParams::lift`): its table in `aux` at LIFT_OFF
    ("LIFT", 1),
    ("LIFT_OFF", 1),
    ("LIFT_N", 1),
    ("LIFT_LO", 1),
    ("LIFT_STEP", 1),
    // process V9's extra Highlights pull next to clipped areas (`FinishParams::edge_pull`): its plane in `aux` at
    // EDGE_OFF, EDGE_W × EDGE_H, read at (origin + pixel + 0.5) · scale − 0.5 and scaled by EDGE_GAIN
    ("EDGE_PULL", 1),
    ("EDGE_OFF", 1),
    ("EDGE_W", 1),
    ("EDGE_H", 1),
    ("EDGE_GAIN", 1),
    ("EDGE_SX", 1),
    ("EDGE_SY", 1),
    ("EDGE_OX", 1),
    ("EDGE_OY", 1),
    // …the surroundings' base on the same grid at EDGE_WIDE_OFF (0 = none), where the picture's pull table (on the
    // LIFT grid, at EDGE_TABLE_OFF) is read
    ("EDGE_WIDE_OFF", 1),
    ("EDGE_TABLE_OFF", 1),
    // …whose cells count by how close their tone is to the pixel's own, by a Gaussian of this sigma (EV)
    ("EDGE_READ_TONE", 1),
    // …plus this much of plain nearness, so a pixel with no cell near its tone doesn't take the least-far cell's
    ("EDGE_READ_FLOOR", 1),
    // process V9's highlight lift under a negative Exposure (`FinishParams::negev_lift`): its table (NEGEV_N rows, by
    // headroom under NEGEV_WHITE in steps of NEGEV_STEP, of NEGEV_GAPS columns, by the brightest channel's gap over
    // luminance in steps of NEGEV_GAP_STEP) in `aux` at NEGEV_OFF
    ("NEGEV", 1),
    ("NEGEV_OFF", 1),
    ("NEGEV_N", 1),
    ("NEGEV_STEP", 1),
    ("NEGEV_WHITE", 1),
    ("NEGEV_GAPS", 1),
    ("NEGEV_GAP_STEP", 1),
    ("CLAR", 1),
    ("TEX", 1),
    ("DEHAZE", 1),
    ("DEHAZE_V3", 1),
    ("SHARPEN_V4", 1),
    // process V5: over-range colours keep hue and saturation (`lcc_pipeline::finish::hue_clip`)
    ("HUE_CLIP", 1),
    // process V8: 8-bit output is ordered-dithered (`lcc_pipeline::finish::bayer_dither_offset`)
    ("DITHER", 1),
    // process V9: bright colours take the tone curve per channel (`lcc_pipeline::finish::top_shoulder`)
    ("TOP_SHOULDER", 1),
    ("SHARPEN", 1),
    ("SHARPEN_MASK", 1),
    ("SHARPEN_LIMIT", 1),
    ("SHARPEN_FADE", 1),
    ("SHP_OFF", 1),
    ("HAS_SHP", 1),
    ("HAS_CLAR", 1),
    ("HAS_TEX", 1),
    ("HAS_DARK", 1),
    ("HAS_CHROMA", 1),
    ("OPS_IDENTITY", 1),
    ("VIBRANCE", 1),
    ("SATURATION", 1),
    ("MIXER", 1),
    ("MIX_HUE", 8),
    ("MIX_SAT", 8),
    ("MIX_LUM", 8),
    // process V9's measured Color Mixer (`ColorOps::measured`): its block in `aux` at MIXT_OFF
    ("MIXT", 1),
    ("MIXT_OFF", 1),
    ("NPC", 1),
    ("PC", 8 * POINT_WORDS),
    ("BW", 1),
    ("BW_MIX", 8),
    ("GRADING", 1),
    ("WHEELS", 12),
    ("BLENDING", 1),
    ("BALANCE", 1),
    ("SPLIT", 1),
    ("SPLIT_K", lcc_pipeline::split_v9::SPLIT_WORDS),
    ("SPLIT_ZSH", 61),
    ("SPLIT_ZHL", 61),
    ("SPLIT_ZMIX", 61),
    ("SKIN", 1),
    ("BANDH", 8),
    // the colourist primaries (`PrimariesK`): the Color wheels then the Log wheels (rgb + master
    // each), the Log wheels' range boundaries
    ("PRIMARIES", 1),
    ("PRIM_LIFT", 4),
    ("PRIM_GAMMA", 4),
    ("PRIM_GAIN", 4),
    ("PRIM_OFFSET", 4),
    ("PRIM_SHADOW", 4),
    ("PRIM_MIDTONE", 4),
    ("PRIM_HIGHLIGHT", 4),
    ("PRIM_LOG_OFFSET", 4),
    ("PRIM_LOW", 1),
    ("PRIM_HIGH", 1),
    // the HSL curves (`HslK`): the hue, saturation and luminance LUTs over the hue circle
    ("HSL", 1),
    ("HSL_LUT", HSL_WORDS),
    ("VIG", 1),
    ("VIG_AMOUNT", 1),
    ("VIG_START", 1),
    ("VIG_WIDTH", 1),
    ("VIG_ASPECT_MIX", 1),
    ("VIG_POWER", 1),
    ("VIG_HL", 1),
    ("VIG_STYLE", 1),
    // the window's offset in the whole output and the whole output's size (pixels)
    ("VIG_VIEW", 4),
    ("GRAIN", 1),
    // 0 = value noise (processes V1–V6: GRAIN_AMT, GRAIN_SC, GRAIN_ROUGH), 1 = particles (V7)
    ("GRAIN_MODEL", 1),
    ("GRAIN_AMT", 1),
    ("GRAIN_AMP", 1),
    ("GRAIN_INV_R", 1),
    ("GRAIN_S", 1),
    ("GRAIN_INV_S", 1),
    ("GRAIN_SPREAD", 1),
    ("GRAIN_ASPECT", 1),
    ("GRAIN_LUMP", 1),
    ("GRAIN_CLUMP_E", 1),
    ("GRAIN_CLUMP_D", 1),
    ("GRAIN_AREA", 1),
    ("GRAIN_SUPPORT", 1),
    ("GRAIN_SUB_REACH", 1),
    ("GRAIN_CELL", 1),
    ("GRAIN_KAPPA", 1),
    ("GRAIN_K_HH", 1),
    ("GRAIN_K_AH", 1),
    ("GRAIN_AREA_REF", 1),
    ("GRAIN_KAPPA_REF", 1),
    ("GRAIN_K_HH_REF", 1),
    ("GRAIN_K_AH_REF", 1),
    ("GRAIN_HALO", 1),
    ("GRAIN_HALO_IN", 1),
    ("GRAIN_HALO_OUT", 1),
    ("GRAIN_HALO_MEAN", 1),
    ("GRAIN_TILT", 1),
    ("GRAIN_WF", 1),
    ("GRAIN_WN", 1),
    ("GRAIN_NOISE_E", 1),
    ("GRAIN_NOISE_N", 1),
    ("GRAIN_COLOR", 1),
    ("GRAIN_GAMMA", 1),
    ("GRAIN_PPL", 1),
    ("GRAIN_SC", 1),
    ("GRAIN_ROUGH", 1),
    ("GRAIN_SEED", 1),
    ("GRAIN_AFF", 6),
    ("REFINE_SAT", 1),
    ("CALIB", 1),
    ("CALIB_M", 9),
    // process V9: the base tone on each linear ProPhoto channel (`FinishParams::tone_per_channel`), these matrices
    // working space → ProPhoto and back
    ("TONE_PC", 1),
    ("TONE_PC_TO", 9),
    ("TONE_PC_FROM", 9),
    ("SHADOW_TINT", 1),
    ("OUT_M", 9),
    ("OUT_INV", 9),
    ("OUT_Y", 3),
    ("OUT_TRC", 1),
    ("OUT_GAMMA", 1),
    // the HDR render's peak (SDR white = 1) and whether the kernel writes linear floats (`DEEP`)
    ("PEAK", 1),
    ("DEEP", 1),
    ("DEEP_OFF", 1),
    // soft proofing (matrix proofs; `ProofParams`): working → proof RGB, its luma, proof → output, warnings
    ("PROOF", 1),
    ("PROOF_M", 9),
    ("PROOF_Y", 3),
    ("PROOF_OUT", 9),
    ("PROOF_DEST_WARN", 1),
    ("PROOF_DISP_WARN", 1),
    // a LUT profile (`finish`'s `lut`): its table in `aux` at LUT_OFF (size³ rgb, red fastest), amount, domain
    ("LUT", 1),
    ("LUT_OFF", 1),
    ("LUT_N", 1),
    ("LUT_AMT", 1),
    ("LUT_MIN", 3),
    ("LUT_MAX", 3),
    // first row of a band dispatch (the kernel runs over rows Y0.., see `render`)
    ("Y0", 1),
];

/// `(name, index)` of every field (for the WGSL constants).
pub fn finish_fields() -> Vec<(&'static str, usize)> {
    let mut i = 0;
    FIELDS
        .iter()
        .map(|(n, len)| {
            let r = (*n, i);
            i += len;
            r
        })
        .collect()
}

/// Offset of field `name`; `None` (and a debug assertion) for a name not in [`FIELDS`].
pub fn index(name: &str) -> Option<usize> {
    let mut i = 0;
    for (n, len) in FIELDS {
        if *n == name {
            return Some(i);
        }
        i += len;
    }
    debug_assert!(false, "unknown finish field {name}");
    None
}

/// A parameter block being filled by name.
pub struct Block(pub Vec<u32>);

impl Block {
    fn new() -> Block {
        Block(vec![0; FIELDS.iter().map(|f| f.1).sum()])
    }
    fn f(&mut self, name: &str, v: f32) {
        self.u(name, v.to_bits());
    }
    fn u(&mut self, name: &str, v: u32) {
        if let Some(slot) = index(name).and_then(|i| self.0.get_mut(i)) {
            *slot = v;
        }
    }
    fn b(&mut self, name: &str, v: bool) {
        self.u(name, v as u32);
    }
    fn fs(&mut self, name: &str, v: &[f32]) {
        let Some(i) = index(name) else { return };
        for (slot, x) in self.0.iter_mut().skip(i).zip(v) {
            *slot = x.to_bits();
        }
    }
}

/// Which optional planes are bound.
pub struct Present {
    pub clarity: bool,
    pub texture: bool,
    /// The sharpening blur plane (after the texture plane in `tex` when both are present).
    pub sharpen: bool,
    pub dark: bool,
    /// The blurred chromaticity follows the mask planes in the `masks` buffer.
    pub chroma: bool,
}

/// The `finish` kernel's parameter block and auxiliary table (tone LUT | chroma curve | sRGB LUT | curve LUTs |
/// mask terms | local-curve slots | local-curve LUTs) for `fp` with `masks` (their local adjustments' terms) and
/// `curves` (their local Curves, [`lcc_pipeline::finish::local_curve_luts`], in the same order).
///
/// A curve whose length is not [`CURVE_N`] fails the render ([`FailKind::Fatal`]) and returns an
/// empty parameter block with a truncated aux table, which the caller discards (CPU fallback).
pub fn finish_block(fp: &FinishParams, masks: &[[f32; MASK_TERMS]], curves: &[Option<[Lut1; 3]>], present: &Present) -> (Vec<u32>, Vec<f32>) {
    let mut aux: Vec<f32> = fp.tone.lut().to_vec();
    aux.extend_from_slice(fp.tone.chroma_lut());
    let srgb_off = aux.len();
    aux.extend_from_slice(&srgb_lut()[..]);
    let curve_off = aux.len();
    if let Some(c) = &fp.curves {
        for l in c {
            if l.v.len() != CURVE_N as usize {
                // a wrong-length curve would shift every later aux offset and make the kernel read
                // the wrong data. `fail()` records the failure on this thread so the render falls
                // back to the CPU; the empty block returned here is never bound to a pipeline.
                fail(FailKind::Fatal, format!("curve table has {} entries, expected {}", l.v.len(), CURVE_N));
                return (Vec::new(), aux);
            }
            aux.extend_from_slice(&l.v);
        }
    }
    let ppcurve_off = aux.len();
    if let Some(c) = &fp.curves_pp {
        for l in c {
            if l.v.len() != CURVE_N as usize {
                // as for `curves` above: a wrong length would shift every later offset
                fail(FailKind::Fatal, format!("curve table has {} entries, expected {}", l.v.len(), CURVE_N));
                return (Vec::new(), aux);
            }
            aux.extend_from_slice(&l.v);
        }
    }
    let mask_off = aux.len();
    for m in masks {
        aux.extend_from_slice(m);
    }
    // one slot per mask (the kernel reads `NMASK` of them), then each local Curve's three tables
    let lcurve_off = aux.len();
    let mut next = 0.0f32;
    for k in 0..masks.len() {
        match curves.get(k) {
            Some(Some(_)) => {
                aux.push(next);
                next += 1.0;
            }
            _ => aux.push(-1.0),
        }
    }
    for l in curves.iter().take(masks.len()).flatten().flatten() {
        aux.extend_from_slice(&l.v);
    }
    // the LUT profile's table, when it is well formed (size ≥ 2, size³ entries)
    let lut_off = aux.len();
    let lut = fp.lut.as_ref().filter(|(l, _)| l.size >= 2 && l.size.checked_pow(3) == Some(l.data.len()));
    if let Some((l, _)) = lut {
        aux.extend(l.data.iter().flatten());
    }
    // process V9's Highlights / Shadows table
    let lift_off = aux.len();
    let lift = fp.lift.as_ref().filter(|l| l.0.len() == lcc_pipeline::tone_lift::LIFT_N);
    if let Some(l) = lift {
        aux.extend_from_slice(&l.0);
    }
    // process V9's raw Highlights pull (only with the table it adds to, as on the CPU)
    let edge_off = aux.len();
    let edge = fp
        .edge_pull
        .as_ref()
        .filter(|e| lift.is_some() && e.plane.width > 0 && e.plane.height > 0 && e.plane.data.len() == e.plane.width * e.plane.height);
    if let Some(e) = edge {
        aux.extend_from_slice(&e.plane.data);
    }
    let lean = edge.filter(|e| e.wide.data.len() == e.plane.data.len() && e.pull.0.len() == lcc_pipeline::tone_lift::LIFT_N);
    let (wide_off, table_off) = (aux.len(), aux.len() + lean.map_or(0, |e| e.wide.data.len()));
    if let Some(e) = lean {
        aux.extend_from_slice(&e.wide.data);
        aux.extend_from_slice(&e.pull.0);
    }
    // process V9's highlight lift under a negative Exposure
    let negev_off = aux.len();
    let negev =
        fp.negev_lift.as_ref().filter(|l| l.table.len() == lcc_pipeline::negev_lift::N * lcc_pipeline::negev_lift::GAPS && l.white.is_finite());
    if let Some(l) = negev {
        aux.extend_from_slice(&l.table);
    }
    // local Color Mixer and Point Color: one record per mask (each list used only when it lines up
    // with the masks, as on the CPU)
    let points_ok = fp.local_points.len() == masks.len();
    let mixers_ok = fp.local_mixers.len() == masks.len();
    // process V9's measured Color Mixers: the colour-space matrices, then the tables; the global one,
    // then each mask's (its record holds the offset)
    let measured_block = |m: &lcc_pipeline::mixer_measured::Measured, aux: &mut Vec<f32>| {
        let o = aux.len();
        aux.extend_from_slice(&m.matrices());
        aux.extend(m.words());
        o
    };
    let mixt_off = fp.ops.measured.as_deref().map(|m| measured_block(m, &mut aux));
    let local_mixt: Vec<Option<usize>> = (0..masks.len())
        .map(|m| match fp.local_mixers.get(m).and_then(Option::as_ref).filter(|_| mixers_ok) {
            Some(Mixer::Measured(t)) => Some(measured_block(t, &mut aux)),
            _ => None,
        })
        .collect();
    let lpoints = !masks.is_empty() && ((points_ok && !fp.local_points.is_empty()) || (mixers_ok && !fp.local_mixers.is_empty()));
    let lpoint_off = aux.len();
    if lpoints {
        for m in 0..masks.len() {
            let mut rec = [0.0f32; LOCAL_POINT_WORDS];
            let points: &[lcc_pipeline::colorops::PointK] = if points_ok { fp.local_points.get(m).map_or(&[], Vec::as_slice) } else { &[] };
            rec[0] = points.len().min(8) as f32;
            match fp.local_mixers.get(m).and_then(Option::as_ref).filter(|_| mixers_ok) {
                Some(Mixer::Bands(mk)) => {
                    rec[1] = 1.0;
                    for (slot, v) in rec.iter_mut().skip(2).zip(mk.words()) {
                        *slot = v;
                    }
                }
                // the block's offset, bit for bit (an f32 holds integers exactly only up to 2^24)
                Some(Mixer::Measured(_)) => {
                    if let Some(o) = local_mixt.get(m).copied().flatten().and_then(|o| u32::try_from(o).ok()) {
                        rec[1] = 2.0;
                        rec[2] = f32::from_bits(o);
                    }
                }
                None => {}
            }
            for (slot, v) in rec.iter_mut().skip(LOCAL_POINT_SAMPLES).zip(points.iter().take(8).flat_map(|k| k.words())) {
                *slot = v;
            }
            aux.extend_from_slice(&rec);
        }
    }

    let mut p = Block::new();
    p.u("W", fp.w as u32);
    p.u("H", fp.h as u32);
    p.u("NMASK", masks.len() as u32);
    p.u("MASK_OFF", mask_off as u32);
    p.u("SRGB_OFF", srgb_off as u32);
    p.u("CURVE_OFF", curve_off as u32);
    p.b("CURVES", fp.curves.is_some());
    if fp.curves_pp.is_some() {
        let (to_pp, from_pp) = lcc_pipeline::finish::prophoto_matrices();
        p.b("PPCURVES", true);
        p.u("PPCURVE_OFF", ppcurve_off as u32);
        p.fs("PP_M", to_pp.as_flattened());
        p.fs("PP_INV", from_pp.as_flattened());
    }
    p.u("LCURVE_OFF", lcurve_off as u32);
    p.b("LCURVES", next > 0.0);
    p.b("LPOINTS", lpoints);
    p.u("LPOINT_OFF", lpoint_off as u32);
    p.f("REFINE_SAT", fp.refine_sat);
    p.f("GAIN", fp.gain);
    p.f("EV", fp.ev);
    p.f("AIR", fp.air);
    p.f("AIR_PRE", fp.air_pre);
    p.f("HL", fp.hl);
    p.f("SH", fp.sh);
    p.f("CLAR", fp.clar);
    p.f("TEX", fp.tex);
    p.f("DEHAZE", fp.dehaze);
    p.b("DEHAZE_V3", fp.dehaze_v3);
    p.b("SHARPEN_V4", fp.sharpen_v4);
    p.b("HUE_CLIP", fp.hue_clip);
    p.b("DITHER", fp.dither);
    p.b("TOP_SHOULDER", fp.top_shoulder);
    p.f("SHARPEN", fp.sharpen);
    p.f("SHARPEN_MASK", fp.sharpen_mask);
    p.f("SHARPEN_LIMIT", fp.sharpen_limit);
    p.f("SHARPEN_FADE", fp.sharpen_fade);
    p.u("SHP_OFF", if present.texture { (fp.w * fp.h) as u32 } else { 0 });
    p.b("HAS_SHP", present.sharpen);
    p.b("HAS_CLAR", present.clarity);
    p.b("HAS_TEX", present.texture);
    p.b("HAS_DARK", present.dark);
    p.b("HAS_CHROMA", present.chroma);

    let ops = &fp.ops;
    p.b("OPS_IDENTITY", ops.is_identity());
    p.f("VIBRANCE", ops.vibrance);
    p.f("SATURATION", ops.saturation);
    p.b("MIXER", ops.mixer);
    p.fs("MIX_HUE", &ops.hue);
    p.fs("MIX_SAT", &ops.sat);
    p.fs("MIX_LUM", &ops.lum);
    p.b("MIXT", mixt_off.is_some());
    p.u("MIXT_OFF", mixt_off.and_then(|o| u32::try_from(o).ok()).unwrap_or(0));
    p.u("NPC", ops.points.len().min(8) as u32);
    let pc: Vec<f32> = ops.points.iter().take(8).flat_map(|k| k.words()).collect();
    p.fs("PC", &pc);
    p.b("BW", ops.bw.is_some());
    p.fs("BW_MIX", &ops.bw.unwrap_or([0.0; 8]));
    if let Some((wheels, blending, balance)) = &ops.grading {
        p.b("GRADING", true);
        let w: Vec<f32> = wheels.iter().flat_map(|k| [k.a, k.b, k.lum]).collect();
        p.fs("WHEELS", &w);
        p.f("BLENDING", *blending);
        p.f("BALANCE", *balance);
    }
    if let Some(pr) = &ops.primaries {
        p.b("PRIMARIES", true);
        for (name, w) in [
            ("PRIM_LIFT", &pr.lift),
            ("PRIM_GAMMA", &pr.gamma),
            ("PRIM_GAIN", &pr.gain),
            ("PRIM_OFFSET", &pr.offset),
            ("PRIM_SHADOW", &pr.shadow),
            ("PRIM_MIDTONE", &pr.midtone),
            ("PRIM_HIGHLIGHT", &pr.highlight),
            ("PRIM_LOG_OFFSET", &pr.log_offset),
        ] {
            p.fs(name, &[w.rgb[0], w.rgb[1], w.rgb[2], w.master]);
        }
        p.f("PRIM_LOW", pr.low);
        p.f("PRIM_HIGH", pr.high);
    }
    if let Some(hsl) = &ops.hsl {
        p.b("HSL", true);
        p.fs("HSL_LUT", &hsl.words());
    }
    if let Some(k) = &ops.split {
        p.b("SPLIT", true);
        p.fs("SPLIT_K", &k.words());
        p.fs("SPLIT_ZSH", &lcc_pipeline::split_v9::SPLIT_ZONE_SH);
        p.fs("SPLIT_ZHL", &lcc_pipeline::split_v9::SPLIT_ZONE_HL);
        p.fs("SPLIT_ZMIX", &lcc_pipeline::split_v9::SPLIT_MIX);
    }
    p.f("SKIN", ops.skin);
    p.fs("BANDH", lcc_pipeline::colorops::band_hues());

    // also the dither's place in the whole output
    p.fs("VIG_VIEW", &fp.view);
    if let Some(v) = &fp.vig {
        p.b("VIG", true);
        p.f("VIG_AMOUNT", v.amount);
        p.f("VIG_START", v.start);
        p.f("VIG_WIDTH", v.width);
        p.f("VIG_ASPECT_MIX", v.aspect_mix);
        p.f("VIG_POWER", v.power);
        p.f("VIG_HL", v.highlights);
        p.u(
            "VIG_STYLE",
            match v.style {
                VignetteStyle::HighlightPriority => 1,
                VignetteStyle::PaintOverlay => 2,
                VignetteStyle::ColorPriority => 0,
            },
        );
    }
    if let Some(m) = &fp.calib {
        p.b("CALIB", true);
        p.fs("CALIB_M", m.as_flattened());
    }
    if let Some([to, from]) = &fp.tone_per_channel {
        p.b("TONE_PC", true);
        p.fs("TONE_PC_TO", to.as_flattened());
        p.fs("TONE_PC_FROM", from.as_flattened());
    }
    p.f("SHADOW_TINT", fp.shadow_tint);
    p.fs("OUT_M", fp.to_out.as_flattened());
    match lcc_pipeline::finish::inv3(&fp.to_out) {
        Some(m) => p.fs("OUT_INV", m.as_flattened()),
        // singular output matrix (never in practice): no split, as the CPU
        None => p.b("SPLIT", false),
    }
    p.fs("OUT_Y", &fp.out_luma);
    let (trc, gamma) = fp.out_trc.code();
    p.u("OUT_TRC", trc);
    p.f("PEAK", if fp.peak.is_finite() { fp.peak.max(1.0) } else { 1.0 });
    p.f("OUT_GAMMA", gamma);
    if let Some(pp) = fp.proof.as_ref().filter(|pp| pp.table.is_none()) {
        p.b("PROOF", true);
        p.fs("PROOF_M", pp.to_proof.as_flattened());
        p.fs("PROOF_Y", &pp.luma);
        p.fs("PROOF_OUT", pp.proof_to_out.as_flattened());
        p.b("PROOF_DEST_WARN", pp.dest_warning);
        p.b("PROOF_DISP_WARN", pp.display_warning);
    }
    if let Some((l, amount)) = lut {
        p.b("LUT", true);
        p.u("LUT_OFF", lut_off as u32);
        p.u("LUT_N", l.size as u32);
        p.f("LUT_AMT", *amount);
        p.fs("LUT_MIN", &l.domain_min);
        p.fs("LUT_MAX", &l.domain_max);
    }
    if lift.is_some() {
        p.b("LIFT", true);
        p.u("LIFT_OFF", lift_off as u32);
        p.u("LIFT_N", lcc_pipeline::tone_lift::LIFT_N as u32);
        p.f("LIFT_LO", lcc_pipeline::tone_lift::LIFT_LO);
        p.f("LIFT_STEP", lcc_pipeline::tone_lift::LIFT_STEP);
    }
    if let Some(e) = edge {
        p.b("EDGE_PULL", true);
        p.u("EDGE_OFF", edge_off as u32);
        p.u("EDGE_W", e.plane.width as u32);
        p.u("EDGE_H", e.plane.height as u32);
        p.f("EDGE_GAIN", e.gain);
        p.f("EDGE_SX", e.scale.0);
        p.f("EDGE_SY", e.scale.1);
        p.u("EDGE_OX", u32::try_from(e.origin.0).unwrap_or(u32::MAX));
        p.u("EDGE_OY", u32::try_from(e.origin.1).unwrap_or(u32::MAX));
    }
    if lean.is_some() {
        p.u("EDGE_WIDE_OFF", wide_off as u32);
        p.u("EDGE_TABLE_OFF", table_off as u32);
        p.f("EDGE_READ_TONE", lcc_pipeline::hl_edge::READ_TONE);
        p.f("EDGE_READ_FLOOR", lcc_pipeline::hl_edge::READ_FLOOR);
    }
    if let Some(l) = negev {
        p.b("NEGEV", true);
        p.u("NEGEV_OFF", negev_off as u32);
        p.u("NEGEV_N", lcc_pipeline::negev_lift::N as u32);
        p.f("NEGEV_STEP", lcc_pipeline::negev_lift::STEP);
        p.f("NEGEV_WHITE", l.white);
        p.u("NEGEV_GAPS", lcc_pipeline::negev_lift::GAPS as u32);
        p.f("NEGEV_GAP_STEP", lcc_pipeline::negev_lift::GAP_STEP);
    }
    let grain_on = fp.grain.is_some() || fp.particles.is_some();
    if let Some((amt, cell, rough, seed)) = fp.grain {
        p.f("GRAIN_AMT", amt);
        p.f("GRAIN_SC", fp.px_per_long as f32 / cell);
        p.f("GRAIN_ROUGH", rough);
        p.u("GRAIN_SEED", seed);
    }
    if let Some(g) = &fp.particles {
        p.u("GRAIN_MODEL", 1);
        p.f("GRAIN_AMP", g.amp);
        p.f("GRAIN_INV_R", g.inv_r);
        p.f("GRAIN_S", g.s);
        p.f("GRAIN_INV_S", g.inv_s);
        p.f("GRAIN_SPREAD", g.spread);
        p.f("GRAIN_ASPECT", g.aspect);
        p.f("GRAIN_LUMP", g.lump);
        p.f("GRAIN_CLUMP_E", g.clump_extra);
        p.f("GRAIN_CLUMP_D", g.clump_d);
        p.f("GRAIN_AREA", g.area);
        p.f("GRAIN_SUPPORT", g.support);
        p.f("GRAIN_SUB_REACH", g.sub_reach);
        p.f("GRAIN_CELL", g.cell);
        p.f("GRAIN_KAPPA", g.kappa);
        p.f("GRAIN_K_HH", g.k_hh);
        p.f("GRAIN_K_AH", g.k_ah);
        p.f("GRAIN_AREA_REF", g.area_ref);
        p.f("GRAIN_KAPPA_REF", g.kappa_ref);
        p.f("GRAIN_K_HH_REF", g.k_hh_ref);
        p.f("GRAIN_K_AH_REF", g.k_ah_ref);
        p.f("GRAIN_HALO", g.halo);
        p.f("GRAIN_HALO_IN", g.halo_in);
        p.f("GRAIN_HALO_OUT", g.halo_out);
        p.f("GRAIN_HALO_MEAN", g.halo_mean);
        p.f("GRAIN_TILT", g.tilt);
        p.f("GRAIN_WF", g.w_field);
        p.f("GRAIN_WN", g.w_noise);
        p.f("GRAIN_NOISE_E", g.noise_e);
        p.f("GRAIN_NOISE_N", g.noise_n);
        p.f("GRAIN_COLOR", g.color);
        p.f("GRAIN_GAMMA", g.gamma);
        p.f("GRAIN_PPL", g.px_per_long);
        p.u("GRAIN_SEED", g.seed);
    }
    if grain_on {
        p.b("GRAIN", true);
        let long = fp.ow.max(fp.oh);
        let [a, b, c, d, e, f] = fp.out_to_norm.0;
        let (kx, ky) = (fp.ow / long, fp.oh / long);
        p.fs("GRAIN_AFF", &[(a * kx) as f32, (b * ky) as f32, (c * kx) as f32, (d * ky) as f32, (e * kx) as f32, (f * ky) as f32]);
    }
    (p.0, aux)
}
