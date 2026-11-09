//! Processes V1–V8 are released: a photo kept on any of them renders exactly as it did on main before process V9
//! was added. Hashes of 8-bit CPU renders of a procedural picture for every V1–V8 process, three kinds of source
//! and edits covering every tone and colour panel (Basic, region and point curves, HSL, colour grading, calibration,
//! presence), recorded on main 7b2ee8bd9 by running this test there with `V1_V8_GOLDEN_RECORD=1`.

use lcc_develop::{Calibration, ColorAdj, ColorGrading, DevelopSettings, Effects, Hsl, Light, Mixer, Process, ToneCurve, Wheel};
use lcc_geom::Point;
use lcc_pipeline::tone::CameraTone;
use lcc_pipeline::{RenderRequest, SourceInfo, render};
use lcc_raster::Rgb32f;

const W: usize = 160;
const H: usize = 112;

/// A smooth ramp from 8 EV under grey to 4 EV over on top; below, a hue sweep at three levels of brightness.
fn scene() -> Rgb32f {
    Rgb32f::from_fn(W, H, |x, y| {
        let t = x as f32 / (W - 1) as f32;
        if y < H / 3 {
            let v = 0.18 * (-8.0 + 12.0 * t).exp2();
            return [v * (1.0 + 0.2 * t), v, v * (1.2 - 0.3 * t)];
        }
        let ev = [-3.0_f32, -0.5, 1.5][((y - H / 3) * 3 / (H - H / 3)).min(2)];
        let h = t * std::f32::consts::TAU;
        let c = |o: f32| 0.55 + 0.45 * (h + o).cos();
        let v = 0.18 * ev.exp2();
        [v * c(0.0), v * c(-2.094), v * c(2.094)]
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

fn pts(p: &[(f64, f64)]) -> Vec<Point> {
    p.iter().map(|&(x, y)| Point::new(x, y)).collect()
}

fn edits() -> Vec<(&'static str, DevelopSettings)> {
    let d = DevelopSettings::default;
    let l = Light::default;
    let c = ToneCurve::default;
    let hsl = |hue, sat, lum| Hsl { hue, sat, lum };
    let wheel = |hue, sat, lum| Wheel { hue, sat, lum };
    vec![
        ("zero", d()),
        (
            "basic",
            DevelopSettings { light: Light { exposure: 0.6, contrast: 45.0, highlights: -85.0, shadows: 17.0, whites: 19.0, blacks: 9.0 }, ..d() },
        ),
        ("sh-up", DevelopSettings { light: Light { shadows: -60.0, highlights: 80.0, ..l() }, ..d() }),
        ("regions", DevelopSettings { curve: ToneCurve { shadows: 40.0, darks: -100.0, lights: 100.0, highlights: -50.0, ..c() }, ..d() }),
        (
            "regions-split",
            DevelopSettings {
                curve: ToneCurve { darks: 21.0, highlights: -60.0, split_shadows: 10.0, split_mid: 56.0, split_highlights: 90.0, ..c() },
                ..d()
            },
        ),
        ("point-master", DevelopSettings { curve: ToneCurve { master: pts(&[(0.0, 0.08), (0.3, 0.25), (0.7, 0.8), (1.0, 0.95)]), ..c() }, ..d() }),
        (
            "point-rgb",
            DevelopSettings {
                curve: ToneCurve {
                    red: pts(&[(0.0, 0.0), (0.14, 0.03), (0.27, 0.16), (1.0, 1.0)]),
                    green: pts(&[(0.0, 0.0), (0.15, 0.05), (0.32, 0.23), (1.0, 1.0)]),
                    blue: pts(&[(0.0, 0.0), (0.18, 0.09), (0.29, 0.2), (1.0, 1.0)]),
                    ..c()
                },
                ..d()
            },
        ),
        (
            "hsl",
            DevelopSettings {
                mixer: Mixer {
                    red: hsl(46.0, -10.0, 9.0),
                    orange: hsl(-6.0, -19.0, 19.0),
                    yellow: hsl(-62.0, -40.0, -14.0),
                    green: hsl(-53.0, -13.0, -10.0),
                    aqua: hsl(13.0, 0.0, 0.0),
                    blue: hsl(-66.0, -22.0, 5.0),
                    purple: hsl(25.0, -25.0, 0.0),
                    magenta: hsl(25.0, -25.0, 0.0),
                },
                ..d()
            },
        ),
        ("vib-sat", DevelopSettings { color: ColorAdj { vibrance: -7.0, saturation: 12.0 }, ..d() }),
        (
            "grading",
            DevelopSettings {
                grading: ColorGrading {
                    shadows: wheel(18.0, 12.0, 0.0),
                    highlights: wheel(42.0, 9.0, 0.0),
                    midtones: wheel(200.0, 6.0, -5.0),
                    balance: 20.0,
                    ..ColorGrading::default()
                },
                ..d()
            },
        ),
        (
            "calibration",
            DevelopSettings {
                calibration: Calibration {
                    shadows_tint: 5.0,
                    red_hue: 30.0,
                    green_hue: 50.0,
                    blue_hue: -10.0,
                    blue_sat: -22.0,
                    ..Calibration::default()
                },
                ..d()
            },
        ),
        ("presence", DevelopSettings { effects: Effects { texture: 20.0, clarity: -10.0, dehaze: 8.0 }, ..d() }),
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
    for p in Process::ALL.into_iter().filter(|p| *p <= Process::V8) {
        for (sn, info) in sources() {
            for (en, s) in edits() {
                let s = DevelopSettings { process: p.version(), ..s };
                let img = render(&src, &info, &s, &RenderRequest::fit(W, H)).image;
                out.push((format!("{}|{sn}|{en}", p.version().0), hash(&img.data)));
            }
        }
    }
    out
}

include!("process_v1_v8_golden_table.rs");

#[test]
fn v1_to_v8_render_exactly_as_on_main() {
    let got = renders();
    if std::env::var_os("V1_V8_GOLDEN_RECORD").is_some() {
        println!("/// Recorded on main 7b2ee8bd9 (Windows x86_64).\nconst GOLDEN: &[(&str, u64)] = &[");
        for (k, h) in &got {
            println!("    (\"{k}\", 0x{h:016x}),");
        }
        println!("];");
        return;
    }
    if !cfg!(all(windows, target_arch = "x86_64")) {
        lcc_testkit::skip!(System, "the V1–V8 golden hashes were recorded on Windows x86_64");
        return;
    }
    assert_eq!(got.len(), GOLDEN.len());
    let wrong: Vec<_> = got.iter().zip(GOLDEN).filter(|((k, h), (gk, gh))| k != gk || h != gh).map(|((k, _), _)| k.as_str()).collect();
    assert!(wrong.is_empty(), "{} V1–V8 renders changed from main's: {wrong:?}", wrong.len());
}
