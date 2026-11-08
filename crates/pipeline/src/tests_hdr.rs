//! HDR editing: the HDR render ([`OutputDepth::F32Hdr`]) and the SDR rendition every other render uses.

use lcc_develop::DevelopSettings;
use lcc_raster::Rgb32f;

use crate::{DeepSamples, OutputDepth, RenderRequest, SourceInfo, render};

fn sunset() -> Rgb32f {
    let lib = lcc_scenes::demo_library();
    let s = lib.iter().find(|s| s.kind == lcc_scenes::Kind::OceanSunset).map(|s| s.render(240, 160));
    s.unwrap_or_else(|| lib[0].render(240, 160))
}

fn raw() -> SourceInfo {
    SourceInfo { raw: true, ..Default::default() }
}

fn hdr_on(max_ev: f64) -> DevelopSettings {
    let mut s = DevelopSettings::default();
    s.hdr.enabled = true;
    s.hdr.max_ev = max_ev;
    s
}

fn deep(src: &Rgb32f, s: &DevelopSettings, depth: OutputDepth) -> Vec<f32> {
    let req = RenderRequest { depth, ..RenderRequest::fit(240, 240) };
    match render(src, &raw(), s, &req).deep.map(|d| d.samples) {
        Some(DeepSamples::F32(v)) => v,
        other => panic!("expected float samples, got {other:?}"),
    }
}

#[test]
fn hdr_midtones_match_the_sdr_render() {
    let src = sunset();
    let hdr = deep(&src, &hdr_on(3.0), OutputDepth::F32Hdr);
    let sdr = deep(&src, &DevelopSettings::default(), OutputDepth::F32Linear);
    // pixels whose brightest channel is a midtone (bright saturated colours legitimately differ:
    // SDR desaturates them at white, HDR keeps their colour)
    let mut n = 0;
    for (h, s) in hdr.as_chunks::<3>().0.iter().zip(sdr.as_chunks::<3>().0) {
        if s.iter().all(|v| *v < 0.15) {
            for (a, b) in h.iter().zip(s) {
                assert!((a - b).abs() < 2e-3, "{h:?} vs {s:?}");
            }
            n += 1;
        }
    }
    assert!(n > 100, "the scene should have midtones and shadows ({n})");
}

#[test]
fn f32hdr_without_hdr_is_the_sdr_float_render() {
    let src = sunset();
    let s = DevelopSettings::default();
    assert_eq!(deep(&src, &s, OutputDepth::F32Hdr), deep(&src, &s, OutputDepth::F32Linear));
}

#[test]
fn sdr_renders_of_an_hdr_edit_use_the_sdr_rendition() {
    let src = sunset();
    let req = RenderRequest::fit(240, 240);
    // neutral offsets: the edit with HDR off, its highlights rolled off (issue #69)
    let plain = render(&src, &raw(), &DevelopSettings::default(), &req);
    let on = render(&src, &raw(), &hdr_on(3.0), &req);
    let rendition = render(&src, &raw(), &hdr_on(3.0).sdr_rendition(), &req);
    assert_eq!(rendition.image.data, on.image.data);
    // offsets apply to every SDR render
    let mut s = hdr_on(3.0);
    s.hdr.sdr_brightness = 50.0;
    s.hdr.sdr_highlights = -40.0;
    let brighter = render(&src, &raw(), &s, &req);
    let expect = render(&src, &raw(), &s.sdr_rendition(), &req);
    assert_eq!(brighter.image.data, expect.image.data);
    assert_ne!(brighter.image.data, plain.image.data);
    let r = s.sdr_rendition();
    assert!(!r.hdr.enabled && (r.light.exposure - 0.5).abs() < 1e-9 && r.light.highlights == -40.0);
}

/// Issue #69: an HDR edit's SDR rendition compresses the highlights an SDR edit would clip, so a bright
/// ramp keeps distinct values above the old clip point; the same edit with HDR off renders as before.
#[test]
fn sdr_rendition_rolls_highlights_off_instead_of_clipping() {
    // a horizontal ramp from mid-grey to 6 stops over grey
    let (w, h) = (256, 4);
    let mut src = Rgb32f::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = 0.18 * 2f32.powf(6.0 * x as f32 / (w - 1) as f32);
            src.data[y * w + x] = [v, v, v];
        }
    }
    let req = RenderRequest::fit(w, w);
    let mut sdr = DevelopSettings::default();
    sdr.light.exposure = 1.5;
    let mut hdr = sdr.clone();
    hdr.hdr.enabled = true;
    let row = |s: &DevelopSettings| -> Vec<u8> {
        let out = render(&src, &raw(), s, &req).image;
        out.data[..out.width].iter().map(|p| p[1]).collect()
    };
    let (old, new) = (row(&sdr), row(&hdr));
    let first_clip = old.iter().position(|v| *v == 255).expect("the SDR edit clips this ramp");
    let distinct = |r: &[u8]| {
        let mut v = r[first_clip..].to_vec();
        v.dedup();
        v.len()
    };
    assert_eq!(distinct(&old), 1, "SDR edit: flat white past its clip point");
    assert!(distinct(&new) > 10, "HDR edit's SDR rendition keeps gradation: {:?}", &new[first_clip..]);
    assert!(new.windows(2).all(|p| p[1] >= p[0]), "monotone");
    // SDR edits are untouched: no rolloff without HDR, whatever a damaged field says
    let mut damaged = sdr.clone();
    damaged.hdr.rolloff_peak = f32::NAN;
    assert_eq!(row(&damaged), old);
    assert_eq!(sdr.sdr_rendition().hdr.rolloff_peak(), 1.0);
}

// Given an HDR edit dragged in a view whose stage cache also serves the view's own (larger)
// render, the HDR histogram with the cache is the histogram without it, update after update
#[test]
fn a_cached_hdr_histogram_is_the_uncached_one_through_a_drag() {
    let src = std::sync::Arc::new(lcc_scenes::demo_library()[1].render(960, 640));
    let cache = crate::StageCache::default();
    let view = RenderRequest::fit(960, 960);
    let mut s = hdr_on(3.0);
    s.light.highlights = -30.0;
    s.effects.clarity = 20.0;
    s.detail.nr_luminance = 25.0;
    for step in 0..6 {
        s.light.exposure = 0.25 * f64::from(step);
        if step == 3 {
            s.effects.clarity = 35.0; // a slider that feeds the spatial planes
        }
        // (the view renders through the same cache between histograms, as the loupe on the CPU does)
        drop(crate::render_cached(&src, &raw(), &s, &view, &cache));
        let cached = crate::hdr_histogram_cached(&src, &raw(), &s, &view, &cache).expect("an HDR histogram");
        let plain = crate::hdr_histogram(&src, &raw(), &s, &view).expect("an HDR histogram");
        assert!(cached.hdr.is_some());
        assert_eq!(
            (cached.luma, cached.r, cached.g, cached.b, cached.total, cached.hdr),
            (plain.luma, plain.r, plain.g, plain.b, plain.total, plain.hdr),
            "step {step}"
        );
    }
}

// Given an HDR edit dragged with its histogram asked for at draft grade at the drag histogram's
// size, it gets an HDR histogram; a loupe-sized draft (the drag's own update) still does not
#[test]
fn a_small_draft_hdr_histogram_request_returns_one() {
    let src = lcc_scenes::demo_library()[1].render(480, 320);
    let s = hdr_on(3.0);
    let edge = crate::HDR_HISTOGRAM_DRAG_EDGE;
    let small = RenderRequest { quality: crate::Quality::Draft, ..RenderRequest::fit(edge, edge) };
    let h = crate::hdr_histogram(&src, &raw(), &s, &small).expect("a draft drag histogram");
    assert!(h.hdr.is_some());
    let loupe = RenderRequest { quality: crate::Quality::Draft, ..RenderRequest::fit(1600, 1600) };
    assert!(crate::hdr_histogram(&src, &raw(), &s, &loupe).is_none(), "loupe drafts keep their own histogram");
}

/// A raw with a camera tone curve (like a user's profile): a soft shoulder reaching white.
fn camera_raw() -> SourceInfo {
    let curve = crate::tone::CameraTone::new(std::array::from_fn(|i| {
        let x = 0.004 * 1.18f32.powi(i as i32);
        [x, 1.0 - (-2.0 * x).exp()]
    }));
    SourceInfo { raw: true, camera_tone: curve, ..Default::default() }
}

fn with_process(mut s: DevelopSettings, p: lcc_develop::Process) -> DevelopSettings {
    s.process = p.version();
    s
}

/// (HDR-off SDR value, LrC's HDR value) of the same tone, measured on Lightroom Classic 15.5 PQ exports of four
/// Canon R5 C raws (medians over 1.5 M pixels, 7 Nov 2026, `_scratch/lrc-calib-2/hdr1`, `fit/hdr_transfer.py`).
const LRC_RAW_PAIRS: [(f32, f32); 6] = [(0.45, 0.459), (0.55, 0.575), (0.65, 0.771), (0.75, 1.024), (0.85, 1.603), (0.925, 2.976)];

/// V9's HDR tone of a raw is LrC's: each tone's HDR value follows LrC's measured transfer from that tone's
/// HDR-off value within 0.1 EV, and the brightest tones reach the headroom exactly (LrC clips there; V7's
/// shoulder only creeps toward it). Fails under V7, whose HDR highlights are up to 1 EV dimmer.
#[test]
fn v10_hdr_tone_follows_lightroom_from_the_sdr_tone() {
    use crate::finish::base_tone;
    use lcc_develop::Process;
    for info in [camera_raw(), raw()] {
        let hdr_s = with_process(hdr_on(2.3), Process::V9);
        let sdr_s = with_process(DevelopSettings::default(), Process::V9);
        let peak = hdr_s.hdr.peak();
        let (h, s) = (base_tone(&hdr_s, &info, peak), base_tone(&sdr_s, &info, 1.0));
        for &(sv, _) in &LRC_RAW_PAIRS {
            // the scene value whose SDR tone is `sv` (the SDR LUT is monotone)
            let i = s.lut().iter().position(|&v| v >= sv).expect("the SDR curve reaches the value");
            let got = h.lut()[i];
            let m = s.lut().iter().copied().fold(0.0f32, f32::max).min(1.0);
            let want = m * crate::tone::lrc_hdr_value(s.lut()[i] / m, peak / m);
            assert!((got / want).log2().abs() < 0.02, "{info:?}: SDR {} -> HDR {got} (model {want})", s.lut()[i]);
        }
        for &(sv, lrc) in &LRC_RAW_PAIRS {
            let model = crate::tone::lrc_hdr_value(sv, peak);
            assert!((model / lrc).log2().abs() < 0.1, "model {model} vs LrC {lrc} at SDR {sv}");
        }
        let top = *h.lut().last().expect("a table");
        assert_eq!(top, peak, "{info:?}: the brightest tones reach the headroom");
        // the V7 HDR curve of the same edit is dimmer above white (here at SDR 0.85 by more than a third of a stop)
        let v7 = base_tone(&with_process(hdr_on(2.3), Process::V7), &info, peak);
        let i = s.lut().iter().position(|&v| v >= 0.85).expect("the SDR curve reaches 0.85");
        if info.raw {
            assert!((h.lut()[i] / v7.lut()[i]).log2() > 0.33, "{info:?}: V9 {} vs V7 {}", h.lut()[i], v7.lut()[i]);
        }
    }
}

/// V9 changes only the HDR render: SDR edits, and the SDR rendition of an HDR edit, tone exactly as under V7.
#[test]
fn v10_keeps_sdr_renders_and_the_sdr_rendition_of_v7() {
    use crate::finish::base_tone;
    use lcc_develop::Process;
    for info in [camera_raw(), raw(), SourceInfo::default()] {
        for s in [DevelopSettings::default(), hdr_on(3.0)] {
            let sdr = s.sdr_rendition().into_owned();
            let a = base_tone(&with_process(sdr.clone(), Process::V9), &info, 1.0);
            let b = base_tone(&with_process(sdr, Process::V7), &info, 1.0);
            assert_eq!(a.lut(), b.lut(), "{info:?}");
        }
    }
}

/// Under V9 a raw edited with Whites −100 still reaches the headroom in HDR, as in Lightroom Classic (4.9 of a
/// 4.9 peak on R5 C raws), and stays brighter than its SDR curve above the mid tones. Failed before the shoulder
/// unrolled toward the SDR curve's own top: the HDR peak stopped at 1.05.
#[test]
fn v10_hdr_reaches_the_headroom_with_whites_lowered() {
    use crate::finish::base_tone;
    use lcc_develop::Process;
    for info in [camera_raw(), raw()] {
        for whites in [-100.0, -50.0] {
            let mut hdr_s = with_process(hdr_on(2.3), Process::V9);
            hdr_s.light.whites = whites;
            let mut sdr_s = with_process(DevelopSettings::default(), Process::V9);
            sdr_s.light.whites = whites;
            let peak = hdr_s.hdr.peak();
            let (h, s) = (base_tone(&hdr_s, &info, peak), base_tone(&sdr_s, &info, 1.0));
            let top = *h.lut().last().expect("a table");
            assert!((top / peak - 1.0).abs() < 1e-4, "{info:?} whites {whites}: HDR top {top} of {peak}");
            assert!(h.lut().iter().zip(s.lut()).all(|(a, b)| a >= &(b * 0.999)), "{info:?}: never darker than SDR");
        }
    }
}

/// V9 keeps V7's HDR curve for display-referred sources (float TIFFs, JPEGs): it matched LrC within 0.2 EV on
/// float TIFF charts, and LrC's raw tone made them 0.4-0.7 EV too bright above white. Failed when V9 gave every
/// source the raw tone.
#[test]
fn v10_keeps_the_v7_hdr_tone_of_display_referred_sources() {
    use crate::finish::base_tone;
    use lcc_develop::Process;
    let info = SourceInfo::default();
    let s = hdr_on(2.3);
    let peak = s.hdr.peak();
    let a = base_tone(&with_process(s.clone(), Process::V9), &info, peak);
    let b = base_tone(&with_process(s, Process::V7), &info, peak);
    assert_eq!(a.lut(), b.lut());
}

/// The HDR value of any SDR value is finite and between the SDR value and the peak, for hostile inputs too.
#[test]
fn lrc_hdr_value_is_bounded_for_any_input() {
    use crate::tone::lrc_hdr_value;
    for s in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0, 1e-9, 0.3, 0.9, 0.999_999, 1.0, 7.0] {
        for peak in [f32::NAN, f32::INFINITY, -2.0, 0.5, 1.0, 4.92, 32.0] {
            let v = lrc_hdr_value(s, peak);
            let p = if peak.is_finite() { peak.max(1.0) } else { 1.0 };
            assert!(v.is_finite() && (0.0..=p).contains(&v), "{s} {peak} -> {v}");
            if s.is_finite() && s > 0.0 {
                assert!(v >= s.min(p), "{s} {peak} -> {v}");
            }
        }
    }
    // monotone in the SDR value
    let mut last = 0.0;
    for i in 0..=1000 {
        let v = lrc_hdr_value(i as f32 / 1000.0, 16.0);
        assert!(v >= last, "{i}");
        last = v;
    }
}
