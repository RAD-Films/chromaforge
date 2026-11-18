//! Process V9 as released (main's: V8's tone with Calibration Saturation in linear light, #1031) renders exactly as
//! on main: the same procedural pictures as `process_v10_golden`, at process V9, against hashes recorded on
//! `origin/main` b3e884609 (18 Nov 2026, Windows x86_64). A camera tone kept under a profile without a curve is
//! V10's mark only: under V9 it renders as the plain camera tone, as on main (which has no such mark).

use lcc_develop::{DevelopSettings, Light, Process, ToneCurve};
use lcc_geom::Point;
use lcc_pipeline::tone::CameraTone;
use lcc_pipeline::{RenderRequest, SourceInfo, render};
use lcc_raster::Rgb32f;

const W: usize = 256;
const H: usize = 160;

/// Top third: a smooth ramp from 8 EV under grey to 4.5 EV over. Middle third: a sky ramp into a flat clipped
/// patch (all channels at the same level, as a blown window) and a patch with only the green channel clipped.
/// Bottom third: dark, mid and bright patches with a little colour and texture.
fn scene() -> Rgb32f {
    Rgb32f::from_fn(W, H, |x, y| {
        let t = x as f32 / (W - 1) as f32;
        if y < H / 3 {
            let v = 0.18 * (-8.0 + 12.5 * t).exp2();
            return [v * (1.0 + 0.2 * t), v, v * (1.2 - 0.3 * t)];
        }
        if y < 2 * H / 3 {
            const CLIP: f32 = 4.0;
            return if x < W / 2 {
                let v = 0.9 * (CLIP / 0.9).powf(2.0 * t);
                [(0.82 * v).min(CLIP), (0.92 * v).min(CLIP), v.min(CLIP)]
            } else if x < 3 * W / 4 {
                [CLIP, CLIP, CLIP]
            } else {
                let tex = 0.05 * ((x + y) % 5) as f32;
                [2.4 + tex, CLIP, 1.6 + tex]
            };
        }
        let ev = if x < W / 3 {
            -4.5 + 0.02 * (y % 7) as f32
        } else if x < 2 * W / 3 {
            0.3 + 0.01 * (x % 5) as f32
        } else {
            2.5 + 0.5 * t
        };
        let v = 0.18 * ev.exp2();
        [v * 1.1, v, v * 0.85]
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
        ("raw-camera-kept", SourceInfo { raw: true, camera_tone: camera_tone().map(CameraTone::under_default_curve), ..SourceInfo::default() }),
    ]
}

fn pts(p: &[(f64, f64)]) -> Vec<Point> {
    p.iter().map(|&(x, y)| Point::new(x, y)).collect()
}

fn edits() -> Vec<(&'static str, DevelopSettings)> {
    let at = |light: Light, curve: ToneCurve| DevelopSettings { process: Process::V9.version(), light, curve, ..DevelopSettings::default() };
    let l = Light::default;
    let c = ToneCurve::default;
    vec![
        ("zero", at(l(), c())),
        ("hl-100", at(Light { highlights: -100.0, ..l() }, c())),
        ("hl-50", at(Light { highlights: -50.0, ..l() }, c())),
        ("hl+50", at(Light { highlights: 50.0, ..l() }, c())),
        ("sh+60", at(Light { shadows: 60.0, ..l() }, c())),
        ("whites+50", at(Light { whites: 50.0, ..l() }, c())),
        ("whites-50", at(Light { whites: -50.0, ..l() }, c())),
        ("blacks-40", at(Light { blacks: -40.0, ..l() }, c())),
        ("blacks+40", at(Light { blacks: 40.0, ..l() }, c())),
        ("exposure-1.5", at(Light { exposure: -1.5, ..l() }, c())),
        ("exposure-1.5-hl-100", at(Light { exposure: -1.5, highlights: -100.0, ..l() }, c())),
        ("contrast+40", at(Light { contrast: 40.0, ..l() }, c())),
        ("master-curve", at(l(), ToneCurve { master: pts(&[(0.0, 0.0), (0.25, 0.18), (0.75, 0.84), (1.0, 1.0)]), ..c() })),
        ("red-curve", at(l(), ToneCurve { red: pts(&[(0.0, 0.0), (0.5, 0.58), (1.0, 1.0)]), ..c() })),
        ("regions", at(l(), ToneCurve { shadows: 40.0, darks: -30.0, lights: 30.0, highlights: -50.0, ..c() })),
        (
            "busy",
            at(
                Light { exposure: -0.7, contrast: -20.0, highlights: -70.0, shadows: 55.0, whites: -30.0, blacks: 25.0, ..l() },
                ToneCurve { darks: 30.0, lights: -40.0, master: pts(&[(0.0, 0.03), (0.6, 0.62), (1.0, 0.97)]), ..c() },
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

/// Recorded on origin/main b3e884609 (the kept-tone rows: main's plain camera-tone hashes).
const GOLDEN: &[(&str, u64)] = &[
    ("display|zero", 0x72fd572a912efd4b),
    ("display|hl-100", 0x3cb7b778fc41d53f),
    ("display|hl-50", 0x6a4d4894406627d9),
    ("display|hl+50", 0xfaa63997bdda3a8c),
    ("display|sh+60", 0x592710d31f5a8a5c),
    ("display|whites+50", 0xe1827db7e4b6acbf),
    ("display|whites-50", 0xebdceb763f2acc12),
    ("display|blacks-40", 0x455cfeb79b161ca0),
    ("display|blacks+40", 0x9e7bc0962dfb086e),
    ("display|exposure-1.5", 0x24c72dc51bb342f8),
    ("display|exposure-1.5-hl-100", 0x3a2343d1fc391bd4),
    ("display|contrast+40", 0x82677cb1d6734c5e),
    ("display|master-curve", 0x6ab200572856f343),
    ("display|red-curve", 0x93acedd45bc62c20),
    ("display|regions", 0xb8c4d2c8c1d94f08),
    ("display|busy", 0x35e7469a8d6783c1),
    ("raw|zero", 0x6c02a36f9778b26f),
    ("raw|hl-100", 0x01bc9f709c082617),
    ("raw|hl-50", 0xc921721eadaedce7),
    ("raw|hl+50", 0xd5b4de957306bfc0),
    ("raw|sh+60", 0x2380cc3e9d6004ad),
    ("raw|whites+50", 0xba505361a410f6fd),
    ("raw|whites-50", 0x68ecfb8101f77bc1),
    ("raw|blacks-40", 0x4bad2b1533ed2ff2),
    ("raw|blacks+40", 0xc6c7a403bc412b58),
    ("raw|exposure-1.5", 0xe26bb9e7acb3a5d8),
    ("raw|exposure-1.5-hl-100", 0xd702d3fae370e8b6),
    ("raw|contrast+40", 0xf4132262ba66ac81),
    ("raw|master-curve", 0x9e7616bec5f5d3ef),
    ("raw|red-curve", 0x5d28c7a72a37455b),
    ("raw|regions", 0xc2ffd2020519680a),
    ("raw|busy", 0xd8888f746a5cef58),
    ("raw-camera|zero", 0x22ac1df73ced1115),
    ("raw-camera|hl-100", 0x6539598f5d3dd527),
    ("raw-camera|hl-50", 0x5d29bfd88405578a),
    ("raw-camera|hl+50", 0x4fab41f8cd4cca2c),
    ("raw-camera|sh+60", 0xca7aa91de9e85f15),
    ("raw-camera|whites+50", 0x949ac3231aec49a1),
    ("raw-camera|whites-50", 0x4bd9fa3ef81a1ded),
    ("raw-camera|blacks-40", 0x2fe07cbe4d6e175f),
    ("raw-camera|blacks+40", 0x215c31b1567b7ea5),
    ("raw-camera|exposure-1.5", 0x1c0080d34f63feea),
    ("raw-camera|exposure-1.5-hl-100", 0x59e35cfcf8dc7bcb),
    ("raw-camera|contrast+40", 0x7ec22ea9e013081d),
    ("raw-camera|master-curve", 0xbc8c40ee12022b56),
    ("raw-camera|red-curve", 0xcb125520e35e2c3a),
    ("raw-camera|regions", 0x471ba40e6fb4a231),
    ("raw-camera|busy", 0x6bf3c485871f7a82),
    ("raw-camera-kept|zero", 0x22ac1df73ced1115),
    ("raw-camera-kept|hl-100", 0x6539598f5d3dd527),
    ("raw-camera-kept|hl-50", 0x5d29bfd88405578a),
    ("raw-camera-kept|hl+50", 0x4fab41f8cd4cca2c),
    ("raw-camera-kept|sh+60", 0xca7aa91de9e85f15),
    ("raw-camera-kept|whites+50", 0x949ac3231aec49a1),
    ("raw-camera-kept|whites-50", 0x4bd9fa3ef81a1ded),
    ("raw-camera-kept|blacks-40", 0x2fe07cbe4d6e175f),
    ("raw-camera-kept|blacks+40", 0x215c31b1567b7ea5),
    ("raw-camera-kept|exposure-1.5", 0x1c0080d34f63feea),
    ("raw-camera-kept|exposure-1.5-hl-100", 0x59e35cfcf8dc7bcb),
    ("raw-camera-kept|contrast+40", 0x7ec22ea9e013081d),
    ("raw-camera-kept|master-curve", 0xbc8c40ee12022b56),
    ("raw-camera-kept|red-curve", 0xcb125520e35e2c3a),
    ("raw-camera-kept|regions", 0x471ba40e6fb4a231),
    ("raw-camera-kept|busy", 0x6bf3c485871f7a82),
];

#[test]
fn v9_renders_as_on_main() {
    let got = renders();
    if std::env::var_os("V9_GOLDEN_RECORD").is_some() {
        for (k, h) in &got {
            println!("    (\"{k}\", 0x{h:016x}),");
        }
        return;
    }
    // (float rounding in the maths library can differ by a level elsewhere; the hashes are this platform's)
    if !cfg!(all(windows, target_arch = "x86_64")) {
        lcc_testkit::skip!(System, "the V9 golden hashes were recorded on Windows x86_64");
        return;
    }
    assert_eq!(got.len(), GOLDEN.len());
    let wrong: Vec<_> = got.iter().zip(GOLDEN).filter(|((k, h), (gk, gh))| k != gk || h != gh).map(|((k, _), _)| k.as_str()).collect();
    assert!(wrong.is_empty(), "V9 renders differ from main's: {wrong:?}");
}
