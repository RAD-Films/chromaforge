//! Process V9's raw Highlights − keeps the detail inside bright areas, as Lightroom Classic's does: clouds in a
//! bright sky stay well above the sky between them, and a smooth bright gradient keeps rising, with no flat band
//! and no line where the pull grows steep (`lcc_pipeline::hl_edge`). Lightroom's −100 keeps clouds 11–14 L* above
//! the sky on our own raws; reading the pull at each pixel's own tone left them 5–7 L* above it on a dark frame,
//! and flattened the gradient.

use lcc_develop::{DevelopSettings, Light, Process};
use lcc_pipeline::{RenderRequest, SourceInfo, render};
use lcc_raster::Rgb32f;

const W: usize = 640;
const H: usize = 480;
/// The sky's tone and the clouds' over it (scene linear; the camera curve's white sits near the clouds).
const SKY: f32 = 1.6;
const CLOUD: f32 = 3.0;

/// A dark frame (a dim room under a window, as most of our clouded skies are): the top 64 rows a sky with round
/// clouds (a raised cosine 32 px across, every 64 px), 32 rows below it a smooth ramp from the sky's tone to the
/// clouds', and dark ground (four fifths of the frame) below.
fn scene() -> Rgb32f {
    Rgb32f::from_fn(W, H, |x, y| {
        let v = if y < 64 {
            let (cx, cy) = ((x % 64) as f32 - 32.0, y as f32 - 32.0);
            let r = (cx * cx + cy * cy).sqrt() / 16.0;
            let c = if r < 1.0 { 0.5 + 0.5 * (std::f32::consts::PI * r).cos() } else { 0.0 };
            SKY * (CLOUD / SKY).powf(c)
        } else if y < 96 {
            SKY * (CLOUD / SKY).powf(x as f32 / (W - 1) as f32)
        } else {
            0.004
        };
        // a blue sky: red and green a little under blue
        [0.82 * v, 0.92 * v, v]
    })
}

fn render_at(highlights: f64) -> Vec<f32> {
    let s = DevelopSettings { process: Process::V9.version(), light: Light { highlights, ..Light::default() }, ..DevelopSettings::default() };
    let info = SourceInfo { raw: true, ..SourceInfo::default() };
    let out = render(&scene(), &info, &s, &RenderRequest::fit(W, H)).image;
    assert_eq!((out.width, out.height), (W, H));
    // L* of each pixel
    out.data
        .iter()
        .map(|p| {
            let lin = |v: u8| {
                let v = f32::from(v) / 255.0;
                if v <= 0.040_45 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
            };
            let y = 0.2126 * lin(p[0]) + 0.7152 * lin(p[1]) + 0.0722 * lin(p[2]);
            if y > 0.008_856 { 116.0 * y.cbrt() - 16.0 } else { 903.3 * y }
        })
        .collect()
}

/// The mean L* over `rows` × `cols`.
fn mean(l: &[f32], rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> f32 {
    let n = rows.len() * cols.len();
    rows.flat_map(|y| cols.clone().map(move |x| (x, y))).map(|(x, y)| l[y * W + x]).sum::<f32>() / n as f32
}

#[test]
fn clouds_stay_above_the_sky_at_highlights_minus_100() {
    let l = render_at(-100.0);
    // cloud centres (32 ± 4 px in each 64 px cell) against the sky between them (the cells' corners)
    let (mut cloud, mut sky) = (0.0, 0.0);
    for cx in [2usize, 4, 6, 8] {
        cloud += mean(&l, 28..36, cx * 64 + 28..cx * 64 + 36);
        sky += mean(&l, 2..8, cx * 64 + 2..cx * 64 + 8);
    }
    let step = (cloud - sky) / 4.0;
    assert!(step >= 9.0, "clouds only {step:.1} L* above the sky at −100");
    // and the pull still darkens the sky
    let l0 = render_at(0.0);
    assert!(mean(&l, 2..8, 2..8) < mean(&l0, 2..8, 2..8) - 5.0, "the sky is pulled");
}

#[test]
fn a_bright_gradient_keeps_rising_at_highlights_minus_100() {
    let l = render_at(-100.0);
    // the ramp's middle rows, in 16 px columns: each brighter than the last, with no flat band
    let cols: Vec<f32> = (0..W / 16).map(|i| mean(&l, 72..88, i * 16..i * 16 + 16)).collect();
    let rises: Vec<f32> = cols.windows(2).map(|w| w[1] - w[0]).collect();
    // reading the pull at each pixel's own tone squeezed the whole 0.9 EV ramp into 1 L*
    let total = cols.last().copied().unwrap_or(0.0) - cols.first().copied().unwrap_or(0.0);
    assert!(total > 2.5, "the gradient flattens: it rises by only {total:.2} L* ({cols:?})");
    let least = rises.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(least > 0.03, "the gradient stalls: it rises by only {least:.2} L* over a 16 px step ({cols:?})");
    // and no step stands out against its neighbours (a line across the gradient)
    let worst = rises.windows(3).map(|w| w[1] - 0.5 * (w[0] + w[2])).fold(0.0f32, |a, d| a.max(d.abs()));
    assert!(worst < 1.0, "a step of {worst:.2} L* stands out in the gradient");
}
