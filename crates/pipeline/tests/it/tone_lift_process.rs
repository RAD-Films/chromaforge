//! Process V8's global Highlights and Shadows follow Lightroom Classic's: one curve per picture, shaped by the
//! picture's histogram (`lcc_pipeline::tone_lift`). Older processes keep the fixed response they had.

use lcc_develop::{DevelopSettings, Light, Process};
use lcc_pipeline::tone_lift::{Lift, Tones};
use lcc_pipeline::{RenderRequest, SourceInfo, render};
use lcc_raster::Rgb32f;

const W: usize = 360;
const H: usize = 240;

/// A display-referred picture: a dark patch (EV `dark` over grey) in the middle of a field at EV `field`.
fn picture(dark: f32, field: f32) -> Rgb32f {
    let lin = |ev: f32| 0.18 * ev.exp2();
    Rgb32f::from_fn(W, H, |x, y| {
        let inside = (120..240).contains(&x) && (80..160).contains(&y);
        [lin(if inside { dark } else { field }); 3]
    })
}

fn settings(p: Process, shadows: f64, highlights: f64) -> DevelopSettings {
    DevelopSettings { process: p.version(), light: Light { shadows, highlights, ..Light::default() }, ..DevelopSettings::default() }
}

/// The green value (0–255) at the middle of the dark patch.
fn centre(src: &Rgb32f, s: &DevelopSettings) -> f64 {
    let out = render(src, &SourceInfo::default(), s, &RenderRequest::fit(W, H)).image;
    f64::from(out.data[(H / 2) * W + W / 2][1])
}

fn srgb8(lin: f64) -> f64 {
    let v = lin.clamp(0.0, 1.0);
    255.0 * if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

#[test]
fn v8_shadows_follow_the_pictures_own_curve() {
    let src = picture(-4.0, 1.0);
    let got = centre(&src, &settings(Process::V8, 100.0, 0.0));
    // the curve for this picture's tones (a sixth at -4 EV, the rest at +1 EV), at the patch's tone
    let tones = Tones::new((0..W * H).map(|i| if (i % W) / 120 == 1 && (80..160).contains(&(i / W)) { -4.0 } else { 1.0 })).unwrap();
    let want = srgb8(0.18 * f64::from((-4.0 + Lift::new(&tones, 1.0, 0.0, None).at_both(-4.0)).exp2()));
    assert!((got - want).abs() <= 2.0, "V8 Shadows +100 on the dark patch: {got} (the curve says {want:.1})");
}

#[test]
fn v8_depends_on_the_rest_of_the_picture_and_older_processes_do_not() {
    // the same dark patch in a bright and in a dim picture
    let (bright, dim) = (picture(-4.0, 1.5), picture(-4.0, -1.5));
    let v7 = settings(Process::V7, 100.0, 0.0);
    assert_eq!(centre(&bright, &v7), centre(&dim, &v7), "V7's Shadows is a fixed response of the tone");
    let v8 = settings(Process::V8, 100.0, 0.0);
    let (b, d) = (centre(&bright, &v8), centre(&dim, &v8));
    assert!((b - d).abs() >= 2.0, "V8's curve is the picture's own: {b} in a bright picture, {d} in a dim one");
    // and V7 renders as it did, not with the V8 curve
    assert_ne!(centre(&bright, &v7), b);
}

#[test]
fn v8_highlights_pull_a_bright_area_down() {
    let src = picture(1.8, -2.0);
    let flat = centre(&src, &settings(Process::V8, 0.0, 0.0));
    let down = centre(&src, &settings(Process::V8, 0.0, -100.0));
    let up = centre(&src, &settings(Process::V8, 0.0, 100.0));
    assert!(down < flat - 8.0, "Highlights -100: {down} vs {flat}");
    assert!(up >= flat, "Highlights +100: {up} vs {flat}");
}

#[test]
fn v8_with_hostile_pictures_renders() {
    for v in [0.0, f32::NAN, f32::INFINITY, 1e30] {
        let src = Rgb32f::filled(48, 32, [v; 3]);
        let out = render(&src, &SourceInfo::default(), &settings(Process::V8, 100.0, -100.0), &RenderRequest::fit(48, 32)).image;
        assert_eq!(out.data.len(), 48 * 32);
    }
}

fn whites(p: Process, whites: f64) -> DevelopSettings {
    DevelopSettings { process: p.version(), light: Light { whites, ..Light::default() }, ..DevelopSettings::default() }
}

/// The green value at the middle of the patch of a raw.
fn raw_centre(src: &Rgb32f, s: &DevelopSettings) -> f64 {
    let info = SourceInfo { raw: true, ..SourceInfo::default() };
    let out = render(src, &info, s, &RenderRequest::fit(W, H)).image;
    f64::from(out.data[(H / 2) * W + W / 2][1])
}

#[test]
fn v8_raw_whites_lifts_a_dim_frame_more_and_a_bright_frame_less_than_v7() {
    // Lightroom's raw Whites +100 depends on how far the picture's brightest tones sit under white
    let dim = picture(-3.0, -2.0);
    let (flat, v7, v8) =
        (raw_centre(&dim, &whites(Process::V8, 0.0)), raw_centre(&dim, &whites(Process::V7, 100.0)), raw_centre(&dim, &whites(Process::V8, 100.0)));
    assert!(v8 - flat > (v7 - flat) + 8.0, "a dim raw: V8 lifts {flat} to {v8}, V7 to {v7}");
    let bright = picture(-1.0, 4.0); // its field shows as white
    let (flat, v7, v8) = (
        raw_centre(&bright, &whites(Process::V8, 0.0)),
        raw_centre(&bright, &whites(Process::V7, 100.0)),
        raw_centre(&bright, &whites(Process::V8, 100.0)),
    );
    assert!(v8 > flat && v8 - flat < v7 - flat, "a bright raw: V8 lifts {flat} to {v8}, less than V7's {v7}");
    // Whites down and older processes keep the tone map's response
    assert_eq!(raw_centre(&bright, &whites(Process::V8, -100.0)), raw_centre(&bright, &whites(Process::V7, -100.0)));
}

/// A wall at `dark` EV over grey on the left half, a window at `bright` EV on the right, `w` × `w / 2`.
fn wall_and_window(w: usize, dark: f32, bright: f32) -> Rgb32f {
    let lin = |ev: f32| 0.18 * ev.exp2();
    Rgb32f::from_fn(w, w / 2, |x, _| [lin(if x < w / 2 { dark } else { bright }); 3])
}

/// The change (EV of linear green) `s` makes against `zero` along the middle row.
fn row_change(src: &Rgb32f, s: &DevelopSettings, zero: &DevelopSettings) -> Vec<f64> {
    let (w, h) = (src.width, src.height);
    let row = |s: &DevelopSettings| {
        let out = render(src, &SourceInfo::default(), s, &RenderRequest::fit(w, h)).image;
        (0..w).map(|x| out.data[(h / 2) * w + x][1]).collect::<Vec<_>>()
    };
    let lin = |v: u8| {
        let v = f64::from(v) / 255.0;
        if v <= 0.040_45 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    row(s).iter().zip(row(zero)).map(|(a, b)| ((lin(*a) + 1e-4) / (lin(b) + 1e-4)).log2()).collect()
}

/// Lightroom's Shadows and Highlights ease off at an edge between their regions and never spread across it (its
/// change maps at full resolution, round 15, Nov 2026): on a dark wall next to a bright window Shadows +100 lifts the
/// wall's last 4–8 px (of 6000) about half as much and the window not at all; Highlights −100 leaves the wall alone
/// and pulls the window's first 4–8 px less. (Read from smaller change maps it looked like a spill over a few percent
/// of the long edge; at full resolution that was the tone curve, not the edge.)
#[test]
fn v8_shadows_and_highlights_ease_off_at_an_edge_like_lightrooms() {
    let w = 2400;
    let e = w / 2;
    let src = wall_and_window(w, -4.0, 1.5);
    let zero = settings(Process::V8, 0.0, 0.0);
    // Shadows +100: the wall's last pixels lift less, back to the full lift within a few px; the window gets none
    let s = row_change(&src, &settings(Process::V8, 100.0, 0.0), &zero);
    let (near, far) = (s[e - 1], s[e - 60]);
    assert!(far > near + 0.3, "Shadows lifts the wall's last pixel less: {near:.2} EV there, {far:.2} far away");
    assert!((s[e - 8] - far).abs() < 0.1, "and is back 8 px from the edge: {:.2} vs {far:.2}", s[e - 8]);
    assert!(s[e..].iter().all(|v| v.abs() < 0.1), "the window gets no lift: {:.2}", s[e..].iter().fold(0.0f64, |m, v| m.max(v.abs())));
    // Highlights −100: the wall keeps its change up to the edge; the window's first pixels are pulled less
    let h = row_change(&src, &settings(Process::V8, 0.0, -100.0), &zero);
    assert!((h[e - 1] - h[e - 60]).abs() < 0.02, "Highlights doesn't spread onto the wall: {:.2} at the edge, {:.2} far", h[e - 1], h[e - 60]);
    let (near, far) = (h[e], h[e + 60]);
    assert!(near > far + 0.5, "Highlights pulls the window's first pixel less: {near:.2} EV there, {far:.2} far away");
    assert!((h[e + 16] - far).abs() < 0.15, "and is back 16 px from the edge: {:.2} vs {far:.2}", h[e + 16]);
}

/// A raw radial ramp: `ev0` EV over grey at the centre, falling `per_px` EV per pixel outwards (a sky around a sun).
fn radial_ramp(w: usize, h: usize, ev0: f32, per_px: f32) -> Rgb32f {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    Rgb32f::from_fn(w, h, |x, y| [0.18 * (ev0 - per_px * (x as f32 - cx).hypot(y as f32 - cy)).exp2(); 3])
}

/// The largest step (between neighbouring pixels) in the slope of output against input tone along the ramp's middle
/// row from the centre outwards, where the input is between `lo` and `hi` EV and the output under `top`: a band
/// (a kink in the tone response) is a step there; a smooth response changes its slope a little every pixel.
fn worst_slope_step(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, ev0: f32, per_px: f32, (lo, hi): (f32, f32), top: f32) -> (f32, f32) {
    let (w, h) = (src.width, src.height);
    let req = RenderRequest { depth: lcc_pipeline::OutputDepth::F32Linear, ..RenderRequest::fit(w, h) };
    let out = render(src, info, s, &req);
    let Some(lcc_pipeline::DeepSamples::F32(px)) = out.deep.map(|d| d.samples) else { panic!("no float render") };
    let lum = |x: usize| {
        let i = 3 * ((h / 2) * w + x);
        0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
    };
    let pts: Vec<(f32, f32)> = (w / 2..w)
        .map(|x| (ev0 - per_px * (x - w / 2) as f32, lum(x)))
        .filter(|(e, y)| (lo..=hi).contains(e) && *y > 1e-4 && *y < top)
        .map(|(e, y)| (e, y.log2()))
        .collect();
    let slopes: Vec<f32> = pts.windows(2).map(|p| (p[1].1 - p[0].1) / (p[1].0 - p[0].0)).collect();
    let mut worst = (0.0f32, 0.0f32);
    for (i, p) in slopes.windows(2).enumerate() {
        let d = (p[1] - p[0]).abs();
        if d > worst.0 {
            worst = (d, pts[i + 1].0);
        }
    }
    worst
}

/// No bands: a smooth raw ramp from far past white down into the shadows keeps a smooth tone response through
/// Highlights and Shadows at both ends of the sliders, under the camera's own curve in Lightroom's default look.
/// (Before: the camera curve and the default look were read linearly between knots, and the Highlights table was
/// put in order by a running maximum: each left a slope step that showed as a ring around a sun.)
#[test]
fn v8_a_smooth_ramp_shows_no_bands() {
    let (w, h, ev0, per_px) = (900, 300, 4.0, 1.0 / 40.0);
    let src = radial_ramp(w, h, ev0, per_px);
    let tone = lcc_pipeline::tone::CameraTone::default_raw().map(|c| c.under_default_curve());
    let info = SourceInfo { raw: true, camera_tone: tone, ..SourceInfo::default() };
    let shoulder = {
        let zero = settings(Process::V8, 0.0, 0.0);
        let req = RenderRequest { depth: lcc_pipeline::OutputDepth::F32Linear, ..RenderRequest::fit(w, h) };
        let Some(lcc_pipeline::DeepSamples::F32(px)) = render(&src, &info, &zero, &req).deep.map(|d| d.samples) else { panic!("no float render") };
        let x = w / 2 + ((ev0 - 2.5) / per_px) as usize;
        let i = 3 * ((h / 2) * w + x);
        0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
    };
    for (sh, hl) in [(0.0, 0.0), (100.0, 0.0), (0.0, -100.0), (100.0, -100.0), (-100.0, 100.0)] {
        // (shown under where the camera curve's shoulder joins its roll-off, 2.5 EV over grey with the sliders at 0:
        // that join is the tone map's own work, not this one's)
        // (and from 0.1 EV, 4 px, under the cone's tip: the narrow base Highlights' past-white pull is read at rounds
        // a point off over about its radius, so the tip itself takes a little less pull, which is no band)
        let (step, at) = worst_slope_step(&src, &info, &settings(Process::V8, sh, hl), ev0, per_px, (-6.0, ev0 - 0.1), shoulder);
        assert!(step < 0.03, "Shadows {sh}, Highlights {hl}: the tone response's slope steps by {step:.3} at {at:+.2} EV");
    }
}

/// A raw's dark room (`dark` EV over grey) with a window (`side` px square) in the middle: a view of `bright` EV
/// patches (up to half an EV dimmer, varying from pixel to pixel as a sky does, so no sensor clip piles up) among mid-grey
/// ones, 4 px each (sky between leaves).
fn room_and_window(side: usize, dark: f32, bright: f32) -> Rgb32f {
    let lin = |ev: f32| 0.18 * ev.exp2();
    let (x0, y0) = ((W - side) / 2, (H - side) / 2);
    Rgb32f::from_fn(W, H, |x, y| {
        let inside = (x0..x0 + side).contains(&x) && (y0..y0 + side).contains(&y);
        let sky = inside && ((x - x0) / 4 + (y - y0) / 4) % 2 == 0;
        let vary = 0.5 * ((x * 7 + y * 13) % 23) as f32 / 23.0;
        [lin(if sky {
            bright - vary
        } else if inside {
            0.0
        } else {
            dark
        }); 3]
    })
}

/// Highlights −100 brings a window in a dim room down from white, as Lightroom does: read at the narrow base, the
/// pull past white follows the window's outline (read at the wide base, a window among dark bars sat under white there
/// and kept its whites, as on our rounds 9b and 10).
#[test]
fn v8_highlights_bring_a_window_in_a_dim_room_down_from_white() {
    let tone = lcc_pipeline::tone::CameraTone::default_raw().map(|c| c.under_default_curve());
    let info = SourceInfo { raw: true, camera_tone: tone, ..SourceInfo::default() };
    let pane = |side: usize, hl: f64| {
        let src = room_and_window(side, -2.0, 3.5);
        let out = render(&src, &info, &settings(Process::V8, 0.0, hl), &RenderRequest::fit(W, H)).image;
        let (x0, y0) = ((W - side) / 2, (H - side) / 2);
        f64::from(out.data[(y0 + 9) * W + x0 + 9][1])
    };
    for side in [26, 50, 98] {
        let (zero, pulled) = (pane(side, 0.0), pane(side, -100.0));
        assert!(zero > 250.0, "a {side} px window is white at zero sliders: {zero}");
        // (read at the wide base: 157, 218 and 239; at the narrow base 132, 193 and 211)
        assert!(pulled < 225.0, "Highlights -100 brings a {side} px window down: {pulled}");
    }
}
