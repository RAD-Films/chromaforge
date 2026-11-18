//! Process V8 (dither, raw tone roll-off, joint colour NR) is released: a photo kept on V8 renders exactly as it
//! did on main before process V9 (Lightroom Classic's Highlights / Shadows, region curve and display tone) was
//! added. Hashes of the 8-bit CPU renders of procedural pictures, recorded on main 7b2ee8bd9 by running this test
//! there with `V8_GOLDEN_RECORD=1` (it prints the table instead of checking it).

use lcc_develop::{DevelopSettings, Light, Process, ToneCurve};
use lcc_pipeline::tone::CameraTone;
use lcc_pipeline::{RenderRequest, SourceInfo, render};
use lcc_raster::Rgb32f;

const W: usize = 192;
const H: usize = 128;

/// A smooth ramp from 8 EV under grey to 4 EV over on top, dark and bright patches below, a little colour.
fn scene() -> Rgb32f {
    Rgb32f::from_fn(W, H, |x, y| {
        let t = x as f32 / (W - 1) as f32;
        let ev = if y < H / 2 {
            -8.0 + 12.0 * t
        } else if x < W / 3 {
            -4.5 + 0.02 * (y % 7) as f32
        } else if x < 2 * W / 3 {
            0.3 + 0.01 * (x % 5) as f32
        } else {
            3.0 + 0.5 * t
        };
        let v = 0.18 * ev.exp2();
        [v * (1.0 + 0.2 * t), v, v * (1.2 - 0.3 * t)]
    })
}

fn camera_tone() -> Option<CameraTone> {
    CameraTone::new(std::array::from_fn(|i| {
        let x = 0.002 * 1.25f32.powi(i as i32);
        [x, 1.0 - (-2.0 * x).exp()]
    }))
}

fn sources() -> Vec<(&'static str, SourceInfo)> {
    vec![
        ("display", SourceInfo::default()),
        ("raw", SourceInfo { raw: true, ..SourceInfo::default() }),
        ("raw-camera", SourceInfo { raw: true, camera_tone: camera_tone(), ..SourceInfo::default() }),
    ]
}

fn edits() -> Vec<(&'static str, DevelopSettings)> {
    let at = |light: Light, curve: ToneCurve| DevelopSettings { process: Process::V8.version(), light, curve, ..DevelopSettings::default() };
    let l = Light::default;
    let c = ToneCurve::default;
    vec![
        ("zero", at(l(), c())),
        ("sh", at(Light { shadows: 100.0, highlights: -100.0, ..l() }, c())),
        ("sh-up", at(Light { shadows: -60.0, highlights: 80.0, ..l() }, c())),
        ("whites-blacks", at(Light { whites: 60.0, blacks: -40.0, ..l() }, c())),
        ("exposure", at(Light { exposure: 1.0, contrast: 30.0, ..l() }, c())),
        ("regions", at(l(), ToneCurve { shadows: 40.0, darks: -100.0, lights: 100.0, highlights: -50.0, ..c() })),
        (
            "busy",
            at(
                Light { exposure: -0.7, contrast: -20.0, highlights: -70.0, shadows: 55.0, whites: -30.0, blacks: 25.0, ..l() },
                ToneCurve { darks: 30.0, lights: -40.0, ..c() },
            ),
        ),
    ]
}

/// FNV-1a over the 8-bit pixels.
fn hash(data: &[[u8; 4]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data.iter().flatten() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn renders() -> Vec<(String, u64)> {
    let src = scene();
    let mut out = Vec::new();
    for (sn, info) in sources() {
        for (en, s) in edits() {
            let img = render(&src, &info, &s, &RenderRequest::fit(W, H)).image;
            out.push((format!("{sn}|{en}"), hash(&img.data)));
        }
    }
    out
}

/// Recorded on main 7b2ee8bd9 (Windows x86_64; run twice, the same both times).
const GOLDEN: &[(&str, u64)] = &[
    ("display|zero", 0x701219c7e528497a),
    ("display|sh", 0x0734e0e5087a184c),
    ("display|sh-up", 0xf2f047d243121721),
    ("display|whites-blacks", 0x8a6fd93bde8038d0),
    ("display|exposure", 0x6949660cfba34f4e),
    ("display|regions", 0xc349b858b18acccf),
    ("display|busy", 0xd240d27c428e5c7c),
    ("raw|zero", 0x548721c2c60f65fc),
    ("raw|sh", 0x784f4ba5cc1b59f5),
    ("raw|sh-up", 0x98ccb0342871f150),
    ("raw|whites-blacks", 0x66ec832b94b4050d),
    ("raw|exposure", 0x254735bbe218b7fb),
    ("raw|regions", 0xff5675d194587766),
    ("raw|busy", 0x540e6f73384aaff5),
    ("raw-camera|zero", 0x63929abe6897511f),
    ("raw-camera|sh", 0xa7d433b495757395),
    ("raw-camera|sh-up", 0x1dc718eaf78b1dfb),
    ("raw-camera|whites-blacks", 0xcf0679a9a447df0b),
    ("raw-camera|exposure", 0xf00b958864391fc6),
    ("raw-camera|regions", 0x12150174cf3cb6cd),
    ("raw-camera|busy", 0xe611c8e3b12a41d6),
];

#[test]
fn v8_renders_exactly_as_mains_v8() {
    let got = renders();
    if std::env::var_os("V8_GOLDEN_RECORD").is_some() {
        for (k, h) in &got {
            println!("    (\"{k}\", 0x{h:016x}),");
        }
        return;
    }
    // (float rounding in the maths library can differ by a level elsewhere; the hashes are this platform's)
    if !cfg!(all(windows, target_arch = "x86_64")) {
        lcc_testkit::skip!(System, "the V8 golden hashes were recorded on Windows x86_64");
        return;
    }
    assert_eq!(got.len(), GOLDEN.len());
    let wrong: Vec<_> = got.iter().zip(GOLDEN).filter(|((k, h), (gk, gh))| k != gk || h != gh).map(|((k, _), _)| k.as_str()).collect();
    assert!(wrong.is_empty(), "V8 renders changed from main's: {wrong:?}");
}

/// The camera-JPEG tone mark ([`CameraTone::under_default_curve`]) is V9's; a V8 photo ignores it.
#[test]
fn v8_ignores_the_kept_camera_jpeg_tone_mark() {
    let src = scene();
    let info = |t: Option<CameraTone>| SourceInfo { raw: true, camera_tone: t, ..SourceInfo::default() };
    for (en, s) in edits() {
        let a = render(&src, &info(camera_tone()), &s, &RenderRequest::fit(W, H)).image;
        let b = render(&src, &info(camera_tone().map(CameraTone::under_default_curve)), &s, &RenderRequest::fit(W, H)).image;
        assert!(a.data == b.data, "{en}: the mark changed a V8 render");
    }
}
