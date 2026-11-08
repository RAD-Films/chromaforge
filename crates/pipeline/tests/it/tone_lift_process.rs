//! Process V9's global Highlights and Shadows follow Lightroom Classic's: one curve per picture, shaped by the
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
    let got = centre(&src, &settings(Process::V9, 100.0, 0.0));
    // the curve for this picture's tones (a sixth at -4 EV, the rest at +1 EV), at the patch's tone
    let tones = Tones::new((0..W * H).map(|i| if (i % W) / 120 == 1 && (80..160).contains(&(i / W)) { -4.0 } else { 1.0 })).unwrap();
    let want = srgb8(0.18 * f64::from((-4.0 + Lift::new(&tones, 1.0, 0.0, None).at(-4.0)).exp2()));
    assert!((got - want).abs() <= 2.0, "V9 Shadows +100 on the dark patch: {got} (the curve says {want:.1})");
}

#[test]
fn v8_depends_on_the_rest_of_the_picture_and_older_processes_do_not() {
    // the same dark patch in a bright and in a dim picture
    let (bright, dim) = (picture(-4.0, 1.5), picture(-4.0, -1.5));
    let v7 = settings(Process::V7, 100.0, 0.0);
    assert_eq!(centre(&bright, &v7), centre(&dim, &v7), "V7's Shadows is a fixed response of the tone");
    let v9 = settings(Process::V9, 100.0, 0.0);
    let (b, d) = (centre(&bright, &v9), centre(&dim, &v9));
    assert!((b - d).abs() >= 2.0, "V9's curve is the picture's own: {b} in a bright picture, {d} in a dim one");
    // and V7 renders as it did, not with the V9 curve
    assert_ne!(centre(&bright, &v7), b);
}

#[test]
fn v8_highlights_pull_a_bright_area_down() {
    let src = picture(1.8, -2.0);
    let flat = centre(&src, &settings(Process::V9, 0.0, 0.0));
    let down = centre(&src, &settings(Process::V9, 0.0, -100.0));
    let up = centre(&src, &settings(Process::V9, 0.0, 100.0));
    assert!(down < flat - 8.0, "Highlights -100: {down} vs {flat}");
    assert!(up >= flat, "Highlights +100: {up} vs {flat}");
}

#[test]
fn v8_with_hostile_pictures_renders() {
    for v in [0.0, f32::NAN, f32::INFINITY, 1e30] {
        let src = Rgb32f::filled(48, 32, [v; 3]);
        let out = render(&src, &SourceInfo::default(), &settings(Process::V9, 100.0, -100.0), &RenderRequest::fit(48, 32)).image;
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
    let (flat, v7, v9) =
        (raw_centre(&dim, &whites(Process::V9, 0.0)), raw_centre(&dim, &whites(Process::V7, 100.0)), raw_centre(&dim, &whites(Process::V9, 100.0)));
    assert!(v9 - flat > (v7 - flat) + 8.0, "a dim raw: V9 lifts {flat} to {v9}, V7 to {v7}");
    let bright = picture(-1.0, 4.0); // its field shows as white
    let (flat, v7, v9) = (
        raw_centre(&bright, &whites(Process::V9, 0.0)),
        raw_centre(&bright, &whites(Process::V7, 100.0)),
        raw_centre(&bright, &whites(Process::V9, 100.0)),
    );
    assert!(v9 > flat && v9 - flat < v7 - flat, "a bright raw: V9 lifts {flat} to {v9}, less than V7's {v7}");
    // Whites down keeps the tone map's response: V8's, which V9 is built on (its roll-off), not V7's
    assert_eq!(raw_centre(&bright, &whites(Process::V9, -100.0)), raw_centre(&bright, &whites(Process::V8, -100.0)));
}
