//! Process V9: a raw's camera-profile tables (hue/saturation map, look table) apply after the white balance, at the
//! chosen white (DNG spec ch. 6), instead of at load at the as-shot white with the white balance re-balancing the
//! colours they had already moved. A grey that the chosen white makes neutral renders neutral.

use lcc_color::Mat3;
use lcc_develop::{DevelopSettings, WbMode, WbScale};
use lcc_raster::Rgb32f;

use crate::{RenderRequest, SourceInfo, local, render};

const AS_SHOT: (f64, f64) = (5500.0, 0.0);
const CHOSEN: f64 = 3200.0;

/// A hue/saturation map that leaves greys alone and saturates everything else 1.8 times (a strong but ordinary
/// camera-profile table: real calibrations scale saturation up to 1.6–1.9).
fn saturating_table() -> lcc_raw::profile::HsvTable {
    let (h, s) = (6usize, 3usize);
    let mut data: Vec<f64> = Vec::new();
    for _ in 0..h {
        for si in 0..s {
            data.extend_from_slice(&[0.0, if si == 0 { 1.0 } else { 1.8 }, 1.0]);
        }
    }
    lcc_raw::profile::HsvTable::from_tags(&[h as u64, s as u64, 1], &data, 0).expect("valid table")
}

fn tags() -> lcc_raw::ColorData {
    let mut tags = lcc_raw::ColorData {
        illuminant: [17, 21],
        color_matrix: [
            Some(Mat3([[0.9, 0.2, -0.15], [-0.3, 1.25, 0.08], [0.02, -0.12, 0.85]])),
            Some(Mat3([[0.7, 0.3, -0.1], [-0.35, 1.3, 0.1], [0.05, -0.2, 1.0]])),
        ],
        ..Default::default()
    };
    tags.profile.hue_sat_map = [Some(saturating_table()), Some(saturating_table())];
    tags
}

/// A raw source developed for the as-shot white; `deferred`: its tables wait for the white balance (V9 decode).
fn info(deferred: bool) -> SourceInfo {
    let mut tags = tags();
    if !deferred {
        // (a V1–V7 decode applied the tables and handed the pipeline the tags without them)
        tags.profile = Default::default();
    }
    let cc = crate::CameraColor {
        tags,
        developed_for: lcc_color::cct::temp_tint_to_xy(AS_SHOT.0, AS_SHOT.1),
        deferred_tables_gain: deferred.then_some(1.0),
    };
    SourceInfo { raw: true, as_shot_temp: AS_SHOT.0, as_shot_tint: AS_SHOT.1, camera_color: Some(std::sync::Arc::new(cc)), ..Default::default() }
}

fn chosen_white() -> DevelopSettings {
    let mut s = DevelopSettings::for_raw(AS_SHOT.0, AS_SHOT.1);
    (s.wb.mode, s.wb.temp, s.wb.tint, s.wb.scale) = (WbMode::Custom, CHOSEN, 0.0, WbScale::Kelvin);
    s
}

/// The as-shot development of a grey that the chosen white makes neutral (`level` after the white balance).
fn grey_as_shot(level: f32) -> [f32; 3] {
    let m = local::wb_matrix_for(&info(true), &chosen_white()).expect("a white balance away from as shot");
    let m = Mat3(m.map(|r| r.map(f64::from))).inverse().expect("invertible");
    m.apply([f64::from(level); 3]).map(|v| v as f32)
}

/// What a V1–V7 decode left in the source for camera RGB developed to `rgb` at the as-shot white: the tables applied.
fn baked(rgb: [f32; 3]) -> [f32; 3] {
    let t = tags();
    let w = lcc_raw::color::illuminant_weight(&t, lcc_color::cct::temp_tint_to_xy(AS_SHOT.0, AS_SHOT.1));
    lcc_raw::profile::ProfileTables::new(&t.profile, w).expect("tables").apply(rgb, 1.0)
}

fn spread(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) / c[0].min(c[1]).min(c[2]).max(1e-9)
}

#[test]
fn a_grey_the_chosen_white_makes_neutral_stays_neutral_through_the_tables() {
    let s = chosen_white();
    let grey = grey_as_shot(0.18);
    assert!(spread(grey) > 1.3, "the as-shot development is far from neutral: {grey:?}");

    // V9: white balance, then the tables at the chosen white
    let info9 = info(true);
    let mut img = Rgb32f::from_fn(1, 1, |_, _| grey);
    local::white_balance(&mut img, &info9, &s);
    let v9 = img.data[0];
    assert!(spread(v9) < 1.001, "V9 renders the grey neutral: {v9:?}");

    // V1–V7: the tables moved the as-shot colour before the white balance re-balanced it
    let info7 = info(false);
    let mut img = Rgb32f::from_fn(1, 1, |_, _| baked(grey));
    local::white_balance(&mut img, &info7, &s);
    let v7 = img.data[0];
    assert!(spread(v7) > 1.1, "the old order tints the grey (this is the fault the order fixes): {v7:?}");
}

#[test]
fn at_the_as_shot_white_the_deferred_tables_give_what_the_decode_gave() {
    let s = DevelopSettings::for_raw(AS_SHOT.0, AS_SHOT.1);
    // (the last two have a negative channel, as saturated colours through a camera matrix do: the decode applied the
    // tables to them unclamped, so the white balance must not clamp before the tables)
    for c in [[0.3f32, 0.2, 0.1], [0.05, 0.12, 0.3], [0.18, 0.18, 0.18], [0.6, 0.1, 0.4], [-0.03, 0.1, 0.3], [0.4, -0.05, 0.02]] {
        let info9 = info(true);
        let mut img = Rgb32f::from_fn(1, 1, |_, _| c);
        local::white_balance(&mut img, &info9, &s);
        let old = baked(c).map(|v| v.max(0.0));
        for k in 0..3 {
            assert!((img.data[0][k] - old[k]).abs() <= 1e-5 * old[k].abs().max(1.0), "{c:?}: {:?} vs {old:?}", img.data[0]);
        }
    }
}

#[test]
fn the_baseline_exposure_gain_sits_between_the_map_and_the_look_table() {
    // the map sees the pixels without the gain the source carries: a value-indexed look shows where it sits
    let mut t = tags();
    t.profile.look_table = lcc_raw::profile::HsvTable::from_tags(&[1, 2, 2], &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5], 0);
    let gain = 2.0f32;
    let cc = crate::CameraColor {
        tags: t.clone(),
        developed_for: lcc_color::cct::temp_tint_to_xy(AS_SHOT.0, AS_SHOT.1),
        deferred_tables_gain: Some(gain),
    };
    let info9 = SourceInfo { camera_color: Some(std::sync::Arc::new(cc)), ..info(true) };
    let s = DevelopSettings::for_raw(AS_SHOT.0, AS_SHOT.1);
    let c = [0.2f32, 0.15, 0.1];
    let mut img = Rgb32f::from_fn(1, 1, |_, _| c.map(|v| v * gain));
    local::white_balance(&mut img, &info9, &s);
    let w = lcc_raw::color::illuminant_weight(&t, lcc_color::cct::temp_tint_to_xy(AS_SHOT.0, AS_SHOT.1));
    let want = lcc_raw::profile::ProfileTables::new(&t.profile, w).expect("tables").apply(c, gain);
    for k in 0..3 {
        assert!((img.data[0][k] - want[k]).abs() < 1e-5, "{:?} vs {want:?}", img.data[0]);
    }
}

#[test]
fn renders_apply_the_deferred_tables_after_the_white_balance() {
    let s = chosen_white();
    let grey = grey_as_shot(0.18);
    let colour = [0.22f32, 0.12, 0.05];
    let src = Rgb32f::from_fn(16, 8, |x, _| if x < 8 { grey } else { colour });
    let px = |r: &crate::Rendered, x: usize| {
        let p = r.image.data[4 * r.image.width + x];
        [i32::from(p[0]), i32::from(p[1]), i32::from(p[2])]
    };
    let req = RenderRequest::fit(16, 8);
    let v9 = render(&src, &info(true), &s, &req);
    let g = px(&v9, 3);
    assert!((g[0] - g[1]).abs() <= 1 && (g[1] - g[2]).abs() <= 1, "V9: the grey renders neutral: {g:?}");
    // the tables are applied (the colour isn't what it is without them)
    let no_tables = SourceInfo { camera_color: info(false).camera_color, ..info(true) };
    let plain = render(&src, &no_tables, &s, &req);
    let (a, b) = (px(&v9, 12), px(&plain, 12));
    assert!((0..3).any(|k| (a[k] - b[k]).abs() >= 3), "the hue/saturation map moved the colour: {a:?} vs {b:?}");
    // a V1–V7 source (tables baked at the as-shot white) renders the same grey tinted
    let old_src = Rgb32f::from_fn(16, 8, |x, _| baked(if x < 8 { grey } else { colour }));
    let old = render(&old_src, &info(false), &s, &req);
    let o = px(&old, 3);
    assert!((o[0] - o[2]).abs() >= 4, "the old order tints the grey: {o:?}");
}

#[test]
fn deferred_tables_keep_the_scene_linear_stage_on_the_cpu() {
    let s = DevelopSettings::for_raw(AS_SHOT.0, AS_SHOT.1);
    assert!(crate::lin_needs_cpu(&s, &info(true)), "no GPU kernel for the tables: the CPU stage runs");
    assert!(!crate::lin_needs_cpu(&s, &info(false)));
}
