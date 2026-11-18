//! Issue #1174: a DNG's `BaselineExposure` reaches the render whatever gives the photo its starting tone: the file's
//! own `ProfileToneCurve`, the tone fitted to its embedded preview (`dngToneFit`) or none at all.
//!
//! The files are made here: linear raw data, the file's colour model the identity onto Rec.2020, a profile with a
//! look table (an identity one, as Lightroom's Adobe Standard carries a look table and no tone curve) and an embedded
//! preview rendered from the scene at the file's baseline exposure, as a raw converter that honours the tag renders it.

use lcc_color::{REC2020, SRGB, luminance_2020, transfer::linear_to_srgb};
use lcc_develop::DevelopSettings;
use lcc_pipeline::{RenderRequest, tone::CameraTone};
use lcc_raw::profile::{HsvTable, ProfileLook, ToneCurve};

const W: usize = 384;
const H: usize = 256;

/// What the made file carries besides its baseline exposure.
#[derive(Clone, Copy, Default)]
struct Spec {
    /// A `ProfileToneCurve` ([`FILE_CURVE`]).
    curve: bool,
    /// An embedded preview.
    preview: bool,
    /// Closer to a Lightroom-converted DNG: the raw twice the preview's size, a hue/saturation map and a look table
    /// that change brightness by hue (in sRGB-encoded value, as Adobe Standard's look table is).
    realistic: bool,
}

/// The camera's own tone on scene luminance: brighter than the app's default raw curve through the midtones.
fn camera_tone(y: f32) -> f32 {
    1.0 - (-3.0 * y).exp()
}

/// A look table that changes nothing (hue shift 0, saturation and value scale 1).
fn identity_look_table() -> HsvTable {
    HsvTable { hue_divisions: 1, sat_divisions: 2, val_divisions: 1, data: vec![[0.0, 1.0, 1.0]; 2], srgb_value: false }
}

/// The DNG `ProfileToneCurve` of the control group: lifts mid-grey as [`camera_tone`] roughly does.
const FILE_CURVE: [f64; 8] = [0.0, 0.0, 0.1, 0.22, 0.4, 0.66, 1.0, 1.0];

/// A look table (or hue/saturation map) that scales saturation by `sat` and value by 0.85–1.15 around the hue circle.
fn shaping_table(sat: f32, srgb_value: bool) -> HsvTable {
    let hues = 6;
    let data = (0..hues * 2).map(|i| [((i % hues) as f32 - 2.5) * 2.0, sat, 0.85 + 0.06 * (i % hues) as f32]).collect();
    HsvTable { hue_divisions: hues, sat_divisions: 2, val_divisions: 1, data, srgb_value }
}

/// A synthetic DNG: linear sensor data, `baseline` EV of `BaselineExposure` (the sum the reader keeps with any
/// `BaselineExposureOffset`), a look table, and what `spec` adds: the file's tone curve, an embedded preview of the
/// scene at the baseline exposure through the file's tables and then [`camera_tone`] (or the file's curve).
fn dng(baseline: f64, spec: Spec) -> Vec<u8> {
    let Spec { curve, preview, realistic } = spec;
    let scale = if realistic { 2 } else { 1 };
    let (w, h) = (W * scale, H * scale);
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w / scale) as f32, (i / w / scale) as f32);
            let v = 0.01 + 0.18 * (0.5 + 0.5 * (x * 0.045).sin() * (y * 0.06).cos()) + 0.12 * x / W as f32;
            let colour = match (i / w % 2, i % w % 2) {
                (0, 0) => 1.15,
                (1, 1) => 0.85,
                _ => 1.0,
            };
            256 + (v * colour * 15744.0) as u16
        })
        .collect();
    let bytes = crate::tests_xmp::synthetic_dng_of(w, h, "RGGB", data, None, Default::default());
    let mut raw = lcc_raw::decode(&bytes).unwrap();
    let tone_curve = if curve { ToneCurve::from_tag(&FILE_CURVE) } else { None };
    raw.color = lcc_raw::ColorData {
        illuminant: [21, 0],
        color_matrix: [Some(REC2020.from_xyz()), None],
        as_shot_neutral: Some([1.0; 3]),
        baseline_exposure: baseline,
        profile: if realistic {
            ProfileLook {
                hue_sat_map: [Some(shaping_table(1.05, false)), None],
                look_table: Some(shaping_table(1.1, true)),
                tone_curve: tone_curve.clone(),
                ..Default::default()
            }
        } else {
            ProfileLook { look_table: Some(identity_look_table()), tone_curve: tone_curve.clone(), ..Default::default() }
        },
        ..Default::default()
    };
    let mut file = lcc_raw::write_dng(&raw, &Default::default()).unwrap();
    if !preview {
        return file;
    }
    let t = lcc_raw::color::camera_transform(&raw, lcc_raw::color::as_shot_white_xy(&raw));
    let developed = raw.develop(lcc_raw::Method::Bilinear).unwrap();
    let developed = lcc_raster::resample::resize(&developed, W, H, lcc_raster::resample::Filter::Box);
    let to_srgb = REC2020.to_space(&SRGB);
    let gain = 2f32.powf(baseline as f32);
    let xy = lcc_raw::color::as_shot_white_xy(&raw);
    let tables = lcc_raw::profile::ProfileTables::new(&raw.color.profile, lcc_raw::color::illuminant_weight(&raw.color, xy)).unwrap();
    let pixels: Vec<u8> = developed
        .data
        .iter()
        .flat_map(|p| {
            let p = tables.apply(t.matrix.apply_f32(std::array::from_fn(|i| p[i] * t.wb[i])), gain);
            let y = luminance_2020(p).max(1e-6);
            let toned = match &tone_curve {
                Some(c) => c.eval(y.min(1.0)),
                None => camera_tone(y),
            };
            to_srgb.apply_f32(p.map(|v| v * toned / y)).map(|v| (255.0 * linear_to_srgb(v).clamp(0.0, 1.0)).round() as u8)
        })
        .collect();
    let image = lcc_codecs::EncodeImage::new(W as u32, H as u32, 3, lcc_codecs::Samples::U8(&pixels));
    let jpeg = lcc_codecs::encode_jpeg(&image, 98, lcc_codecs::ChromaSubsampling::S444, &Default::default()).unwrap();
    // Append a preview IFD to the little-endian DNG written above.
    let root = u32::from_le_bytes(file[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(file[root..root + 2].try_into().unwrap()) as usize;
    let next = root + 2 + 12 * count;
    let start = file.len() as u32;
    file[next..next + 4].copy_from_slice(&start.to_le_bytes());
    let jpeg_at = start + 2 + 4 * 12 + 4;
    file.extend(4u16.to_le_bytes());
    for (tag, value) in [(256u16, W as u32), (257, H as u32), (0x201, jpeg_at), (0x202, jpeg.len() as u32)] {
        file.extend(tag.to_le_bytes());
        file.extend(4u16.to_le_bytes());
        file.extend(1u32.to_le_bytes());
        file.extend(value.to_le_bytes());
    }
    file.extend(0u32.to_le_bytes());
    file.extend(&jpeg);
    file
}

/// Mean linear luminance of an 8-bit sRGB picture.
fn mean_luminance(pixels: impl Iterator<Item = [u8; 3]>) -> f64 {
    let lin = |v: u8| lcc_color::transfer::srgb_to_linear(f32::from(v) / 255.0);
    let (mut sum, mut n) = (0.0, 0.0);
    for p in pixels {
        // (sRGB and Rec.709 share primaries)
        sum += f64::from(0.2126 * lin(p[0]) + 0.7152 * lin(p[1]) + 0.0722 * lin(p[2]));
        n += 1.0;
    }
    sum / f64::max(n, 1.0)
}

/// The file's render at default settings (the latest process) plus `exposure` EV: its mean luminance, and the
/// starting tone the file got.
fn render(bytes: &[u8], exposure: f64) -> (f64, Option<CameraTone>) {
    let (source, info) = crate::files::load_bytes(bytes, W).unwrap();
    let mut s = DevelopSettings::default();
    s.light.exposure = exposure;
    let image = lcc_pipeline::render(&source, &info, &s, &RenderRequest::fit(W, W)).image;
    (mean_luminance(image.data.iter().map(|p| [p[0], p[1], p[2]])), info.camera_tone)
}

/// The embedded preview's mean luminance.
fn preview_luminance(bytes: &[u8]) -> f64 {
    let (preview, _) = crate::files::load_embedded_preview(bytes, W).expect("a preview");
    // (decoded to linear Rec.2020)
    preview.data.iter().map(|p| f64::from(luminance_2020(*p))).sum::<f64>() / preview.data.len().max(1) as f64
}

fn ev(a: f64, b: f64) -> f64 {
    (a / b).log2()
}

/// `dngToneFit`: a file with a look table, no `ProfileToneCurve` and `BaselineExposure` 0.25 renders as bright as the
/// embedded preview its tone is fitted to.
#[test]
fn dng_tone_fit_render_reaches_the_preview_with_a_baseline_exposure() {
    for realistic in [false, true] {
        let bytes = dng(0.25, Spec { preview: true, realistic, ..Default::default() });
        let (render, tone) = render(&bytes, 0.0);
        assert!(tone.is_some(), "the tone is fitted to the preview (realistic {realistic})");
        let preview = preview_luminance(&bytes);
        eprintln!("dngToneFit (realistic {realistic}): render {render:.4}, preview {preview:.4}, {:+.3} EV", ev(render, preview));
        assert!(
            ev(render, preview).abs() < 0.08,
            "realistic {realistic}: render {render:.4} vs preview {preview:.4}: {:+.3} EV",
            ev(render, preview)
        );
    }
}

/// `none`: without a preview to fit (and no tone curve), the baseline exposure still renders as that much Exposure.
#[test]
fn dng_without_a_tone_renders_its_baseline_exposure() {
    let (with, tone) = render(&dng(0.25, Spec::default()), 0.0);
    assert!(tone.is_none(), "nothing to fit");
    let (plain, _) = render(&dng(0.0, Spec::default()), 0.25);
    eprintln!("none: baseline 0.25 {with:.4}, baseline 0 + Exposure 0.25 {plain:.4}, {:+.3} EV", ev(with, plain));
    assert!(ev(with, plain).abs() < 0.02, "{with:.4} vs {plain:.4}: {:+.3} EV", ev(with, plain));
}

/// The control group: a file with its own `ProfileToneCurve` (`dngProfileCurve`) and a baseline exposure that the
/// profile's `BaselineExposureOffset` cancels renders with the file's curve, at the preview's brightness.
#[test]
fn dng_profile_curve_render_is_unchanged() {
    let bytes = dng(0.0, Spec { curve: true, preview: true, ..Default::default() });
    let (render, tone) = render(&bytes, 0.0);
    let curve = tone.expect("the file's own tone curve");
    let file = ToneCurve::from_tag(&FILE_CURVE).and_then(|c| crate::files::dng_tone_curve(&c)).unwrap();
    assert_eq!(curve, file, "the file's curve, not a fit");
    let preview = preview_luminance(&bytes);
    eprintln!("dngProfileCurve: render {render:.4}, preview {preview:.4}, {:+.3} EV", ev(render, preview));
    assert!(ev(render, preview).abs() < 0.08, "render {render:.4} vs preview {preview:.4}: {:+.3} EV", ev(render, preview));
}
