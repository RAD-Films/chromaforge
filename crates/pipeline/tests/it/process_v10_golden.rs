//! Process V10 (Lightroom Classic's tone: the Highlights pull and clipped-highlight recovery, Shadows / Whites /
//! Blacks, the tone-curve space, near-white colour) is pinned: merges and refactors must leave a V10 render as it
//! is. Hashes of the 8-bit CPU renders of procedural pictures, recorded on `fix/hl-gradient` e93ccbd5f (where this
//! look was process V9, before main's released V9 took that number back) by running this test with
//! `V10_GOLDEN_RECORD=1` (it prints the table instead of checking it). A change that means to move
//! V10 records the table again and says so in its commit.

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
    let at = |light: Light, curve: ToneCurve| DevelopSettings { process: Process::V10.version(), light, curve, ..DevelopSettings::default() };
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

/// Recorded on fix/hl-gradient e93ccbd5f (Windows x86_64; run twice, the same both times).
const GOLDEN: &[(&str, u64)] = &[
    ("display|zero", 0x72fd572a912efd4b),
    ("display|hl-100", 0x75e9905a6e56a76f),
    ("display|hl-50", 0x3eadf84688da0495),
    ("display|hl+50", 0xe000054baaa7b688),
    ("display|sh+60", 0xf64d69e7bc9a1995),
    ("display|whites+50", 0x85cb260e866d09a7),
    ("display|whites-50", 0xe6c0c10bc8230a63),
    ("display|blacks-40", 0x9eb02e32fd3b04ac),
    ("display|blacks+40", 0x258c12b52c088024),
    ("display|exposure-1.5", 0x24c72dc51bb342f8),
    ("display|exposure-1.5-hl-100", 0xf78e0ca9280cb72a),
    ("display|contrast+40", 0x6ea604460630ec27),
    ("display|master-curve", 0x24bc4a5feff04d1f),
    ("display|red-curve", 0xa499c8ff3f61b261),
    ("display|regions", 0x84b168648d264942),
    ("display|busy", 0xf34b2dccd6792919),
    ("raw|zero", 0x187783a0e17fedf9),
    ("raw|hl-100", 0x6df532d35fa3ebdb),
    ("raw|hl-50", 0x54a03d28c150a32b),
    ("raw|hl+50", 0x20319482e3524b11),
    ("raw|sh+60", 0xfe0afa471e660a0c),
    ("raw|whites+50", 0x959fad671acd6f15),
    ("raw|whites-50", 0x2510afcc8979edd4),
    ("raw|blacks-40", 0x00309cac03f752ec),
    ("raw|blacks+40", 0xe85aa7bd64ccf42a),
    ("raw|exposure-1.5", 0x662e00d4c314cf25),
    ("raw|exposure-1.5-hl-100", 0x5bcffdd337ed102a),
    ("raw|contrast+40", 0x8c7b5ab54815a88f),
    ("raw|master-curve", 0x22b1322da6efdd40),
    ("raw|red-curve", 0xc47f5f36c8baa554),
    ("raw|regions", 0xb154eee342a9f055),
    ("raw|busy", 0xe863f569ee1ab4e0),
    ("raw-camera|zero", 0x5cd90d991f9383b8),
    ("raw-camera|hl-100", 0xbb4c578503974750),
    ("raw-camera|hl-50", 0x2cadf88e9154a3b4),
    ("raw-camera|hl+50", 0x7a565b23ac1637fe),
    ("raw-camera|sh+60", 0xc92be2f382798a26),
    ("raw-camera|whites+50", 0xd031f6cc1cbfc0f2),
    ("raw-camera|whites-50", 0xdc60f14fa10354aa),
    ("raw-camera|blacks-40", 0x7e1e7a99e6f0d038),
    ("raw-camera|blacks+40", 0xa9521cb09a2abff9),
    ("raw-camera|exposure-1.5", 0x2ee862d2acf88683),
    ("raw-camera|exposure-1.5-hl-100", 0xcc2ebe346279ee3b),
    ("raw-camera|contrast+40", 0x84be68b5c27f5c44),
    ("raw-camera|master-curve", 0x140e349a4c2d392b),
    ("raw-camera|red-curve", 0x9f1c9f6504f9ba3b),
    ("raw-camera|regions", 0xe56e641814f32c26),
    ("raw-camera|busy", 0x1933cd34873032d8),
    ("raw-camera-kept|zero", 0xa71d48310f3b7a51),
    ("raw-camera-kept|hl-100", 0x7a3c76cbe060ec53),
    ("raw-camera-kept|hl-50", 0x094e345cbf834873),
    ("raw-camera-kept|hl+50", 0x09eea3eec53a638e),
    ("raw-camera-kept|sh+60", 0x505aceeab83c7f70),
    ("raw-camera-kept|whites+50", 0x9febd7ba6e9ac49f),
    ("raw-camera-kept|whites-50", 0x02f85db70aafbd66),
    ("raw-camera-kept|blacks-40", 0x46ea0213a56e939b),
    ("raw-camera-kept|blacks+40", 0x77f96d3f1a8a3d16),
    ("raw-camera-kept|exposure-1.5", 0x360ea0c3456f8c17),
    ("raw-camera-kept|exposure-1.5-hl-100", 0x4f36379c47edd01f),
    ("raw-camera-kept|contrast+40", 0xebfe8938647e3e67),
    ("raw-camera-kept|master-curve", 0x278aeab1d4283cdb),
    ("raw-camera-kept|red-curve", 0x7b65f353af9d6f0b),
    ("raw-camera-kept|regions", 0x40de072fcd95ddae),
    ("raw-camera-kept|busy", 0x46168742c23718c8),
];

#[test]
fn v10_renders_as_pinned() {
    let got = renders();
    if std::env::var_os("V10_GOLDEN_RECORD").is_some() {
        for (k, h) in &got {
            println!("    (\"{k}\", 0x{h:016x}),");
        }
        return;
    }
    // (float rounding in the maths library can differ by a level elsewhere; the hashes are this platform's)
    if !cfg!(all(windows, target_arch = "x86_64")) {
        lcc_testkit::skip!(System, "the V10 golden hashes were recorded on Windows x86_64");
        return;
    }
    assert_eq!(got.len(), GOLDEN.len());
    let wrong: Vec<_> = got.iter().zip(GOLDEN).filter(|((k, h), (gk, gh))| k != gk || h != gh).map(|((k, _), _)| k.as_str()).collect();
    assert!(wrong.is_empty(), "V10 renders changed from the pinned ones: {wrong:?}");
}
