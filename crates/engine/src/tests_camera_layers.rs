//! A camera profile and a look are two layers (ported from OneSlips, LR-PROF-LAYERS): a camera profile with a look of
//! its own (tone curve, look table) develops with that look under the default look ("As profile"); any other look
//! sits on its calibration only, so two looks are never stacked. Profiles are built in code (none is ever a fixture).

use lcc_catalog::{MediaKind, Op, Photo, PhotoId, Source};
use serde_json::json;

use crate::Session;
use crate::media::LoadOptions;

const MAKE: &str = "TESTMAKE";
const MODEL: &str = "LAYERS1";
const WITH_LOOK: &str = "dcp:layers-test-with-look";
const PLAIN: &str = "dcp:layers-test-plain";
const TABLE: &str = "dcp:layers-test-look-table";
const SATMAP: &str = "dcp:layers-test-saturation-map";

fn profile(name: &str, look: bool) -> lcc_raw::dcp::Dcp {
    let m = lcc_raw::Mat3([[0.7, -0.1, -0.05], [-0.5, 1.3, 0.2], [-0.1, 0.25, 0.65]]);
    let mut color = lcc_raw::ColorData { illuminant: [21, 0], color_matrix: [Some(m), None], ..Default::default() };
    if look {
        color.profile.tone_curve = lcc_raw::profile::ToneCurve::from_tag(&[0.0, 0.0, 0.25, 0.42, 1.0, 1.0]);
    }
    lcc_raw::dcp::Dcp { model: format!("{MAKE} {MODEL}"), name: name.into(), copyright: String::new(), color }
}

/// The two test profiles in the decoder's list of imported ones (beside whatever other tests put there).
fn install() -> std::sync::MutexGuard<'static, ()> {
    let lock = crate::camera_profiles::test_library_lock();
    let mut list: Vec<(String, lcc_raw::dcp::Dcp)> = lcc_raw::dcp::library()
        .into_iter()
        .filter(|(k, _)| ![WITH_LOOK, PLAIN, TABLE, SATMAP].contains(&k.as_str()))
        .map(|(k, d)| (k, (*d).clone()))
        .collect();
    list.push((WITH_LOOK.into(), profile("With Look", true)));
    list.push((PLAIN.into(), profile("Plain", false)));
    list.push((TABLE.into(), table_profile()));
    list.push((SATMAP.into(), saturation_map_profile()));
    lcc_raw::dcp::install_library(list);
    lock
}

/// A tiny RGGB DNG of the test camera.
fn dng() -> Vec<u8> {
    let (w, h) = (64usize, 48usize);
    let mut raw = lcc_raw::RawImage {
        format: lcc_raw::RawFormat::Dng,
        width: w,
        height: h,
        cpp: 1,
        data: lcc_raw::RawData::U16((0..w * h).map(|i| 1000 + (i % 97) as u16 * 40).collect()),
        cfa: lcc_raw::Cfa::bayer("RGGB"),
        bits: 14,
        black: lcc_raw::BlackLevel::uniform(512.0),
        white: vec![16000.0],
        active_area: lcc_raw::Rect::new(0, 0, w, h),
        crop: lcc_raw::Rect::new(0, 0, w, h),
        orientation: lcc_raw::Orientation::Normal,
        color: lcc_raw::ColorData {
            illuminant: [21, 0],
            color_matrix: [Some(lcc_raw::Mat3([[0.8, -0.2, -0.05], [-0.45, 1.3, 0.15], [-0.08, 0.22, 0.62]])), None],
            as_shot_neutral: Some([0.5, 1.0, 0.7]),
            ..Default::default()
        },
        wb_multipliers: None,
        linearized: false,
        pixel_aspect: 1.0,
        opcodes: Default::default(),
        metadata: Default::default(),
        lens_metadata: Default::default(),
    };
    raw.metadata.make = Some(MAKE.into());
    raw.metadata.model = Some(MODEL.into());
    lcc_raw::write_dng(&raw, &Default::default()).unwrap()
}

fn raw_photo(s: &mut Session) -> PhotoId {
    let id = s.catalog.alloc_photo_id();
    let mut p = Photo::new(id, Source::File { path: "/nonexistent/LAYERS.DNG".into() }, "LAYERS.DNG", "DNG", 64, 48, "2026-01-01");
    p.kind = MediaKind::Raw;
    p.meta.camera = format!("{MAKE} {MODEL}");
    s.commit("Add", Op::AddPhoto { photo: Box::new(p) }).unwrap();
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    id
}

#[test]
fn a_look_over_a_camera_profile_with_its_own_look_decodes_with_its_calibration_only() {
    let _lock = install();
    let bytes = dng();
    let decode = |look: bool| {
        let options = LoadOptions { camera: WITH_LOOK.into(), camera_look: look, ..Default::default() };
        crate::files::load_bytes_with(&bytes, 64, options).unwrap().1
    };
    assert!(decode(true).camera_tone.is_some(), "As profile: the profile's own tone curve");
    assert!(decode(false).camera_tone.is_none(), "under another look: the calibration only, no tone curve stacked under it");
    assert!(profile("x", true).has_look() && !profile("x", false).has_look());
    let mut identity = profile("x", false);
    identity.color.profile.tone_curve = lcc_raw::profile::ToneCurve::from_tag(&[0.0, 0.0, 1.0, 1.0]);
    assert!(!identity.has_look(), "an identity tone curve is no look");
}

#[test]
fn choosing_a_camera_profile_with_a_look_is_as_profile_and_a_look_keeps_its_calibration() {
    let _lock = install();
    let mut s = Session::new();
    let id = raw_photo(&mut s);
    s.execute("develop.profile", &json!({"id": "lc.vivid"})).unwrap();
    // a camera profile with a look of its own comes with it: the look goes back to the default
    s.execute("develop.profile", &json!({"id": WITH_LOOK})).unwrap();
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!((p.develop.profile.camera.as_str(), p.develop.profile.id.as_str()), (WITH_LOOK, lcc_develop::DEFAULT_LOOK));
    assert!(crate::camera_profiles::as_profile(&p));
    assert!(LoadOptions::for_photo(&p).camera_look, "decoded with the profile's look");
    // a look of the app's keeps the camera profile as the calibration, without its look
    s.execute("develop.profile", &json!({"id": "lc.vivid"})).unwrap();
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!((p.develop.profile.camera.as_str(), p.develop.profile.id.as_str()), (WITH_LOOK, "lc.vivid"));
    assert!(!crate::camera_profiles::as_profile(&p));
    assert!(!LoadOptions::for_photo(&p).camera_look, "the calibration only");
    // a camera profile without a look keeps the look, and a change of look never decodes it again
    s.execute("develop.profile", &json!({"id": PLAIN})).unwrap();
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!((p.develop.profile.camera.as_str(), p.develop.profile.id.as_str()), (PLAIN, "lc.vivid"));
    assert!(LoadOptions::for_photo(&p).camera_look);
    assert!(s.execute("develop.profile", &json!({"id": "dcp:no-such-profile"})).is_err());
}

#[test]
fn settings_from_before_the_two_layers_keep_their_look_on_the_default() {
    let _lock = install();
    // an old edit: the camera profile chosen with develop.cameraProfile, the look left at the default
    let mut s = Session::new();
    let id = raw_photo(&mut s);
    s.execute("develop.cameraProfile", &json!({"id": WITH_LOOK})).unwrap();
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!(p.develop.profile.id, lcc_develop::DEFAULT_LOOK);
    assert!(crate::camera_profiles::as_profile(&p), "renders as it did: with the profile's own look");
    assert!(LoadOptions::for_photo(&p).camera_look);
}

/// An Adobe Standard-style profile: a matrix and a look table, no tone curve.
fn table_profile() -> lcc_raw::dcp::Dcp {
    let mut d = profile("Look Table", false);
    d.color.profile.look_table = lcc_raw::profile::HsvTable::from_tags(&[1, 2, 1], &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 0);
    assert!(d.color.profile.look_table.is_some() && d.has_look());
    d
}

/// A DNG of the test camera with a camera JPEG: a smooth scene developed through the file's own matrix and a bright,
/// contrasty camera curve (`1 - e^(-3y)`), so the DNG tone fit learns a tone well away from the default one.
fn dng_with_camera_jpeg() -> Vec<u8> {
    use lcc_color::{REC2020, SRGB, luminance_2020, transfer::linear_to_srgb};
    let (w, h) = (384usize, 256usize);
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let v = 0.02 + 0.2 * (0.5 + 0.5 * (x * 0.045).sin() * (y * 0.06).cos()) + 0.1 * x / w as f32;
            let colour = match (i / w % 2, i % w % 2) {
                (0, 0) => 1.2,
                (1, 1) => 0.8,
                _ => 1.0,
            };
            256 + (v * colour * 15744.0) as u16
        })
        .collect();
    let bytes = crate::tests_xmp::synthetic_dng_of(w, h, "RGGB", data, None, Default::default());
    let mut raw = lcc_raw::decode(&bytes).unwrap();
    raw.metadata.make = Some(MAKE.into());
    raw.metadata.model = Some(MODEL.into());
    raw.color = lcc_raw::ColorData {
        illuminant: [21, 0],
        color_matrix: [Some(REC2020.from_xyz()), None],
        as_shot_neutral: Some([1.0; 3]),
        ..Default::default()
    };
    let t = lcc_raw::color::camera_transform(&raw, lcc_raw::color::as_shot_white_xy(&raw));
    let developed = raw.develop(lcc_raw::Method::Bilinear).unwrap();
    let to_srgb = REC2020.to_space(&SRGB);
    let pixels: Vec<u8> = developed
        .data
        .iter()
        .flat_map(|p| {
            let p = t.matrix.apply_f32(std::array::from_fn(|i| p[i] * t.wb[i]));
            let y = luminance_2020(p).max(1e-6);
            let gain = (1.0 - (-3.0 * y).exp()) / y;
            to_srgb.apply_f32(p.map(|v| v * gain)).map(|v| (255.0 * linear_to_srgb(v).clamp(0.0, 1.0)).round() as u8)
        })
        .collect();
    let image = lcc_codecs::EncodeImage::new(w as u32, h as u32, 3, lcc_codecs::Samples::U8(&pixels));
    let jpeg = lcc_codecs::encode_jpeg(&image, 98, lcc_codecs::ChromaSubsampling::S444, &Default::default()).unwrap();
    let mut file = lcc_raw::write_dng(&raw, &Default::default()).unwrap();
    // a preview IFD after IFD0 of the little-endian DNG written above
    let root = u32::from_le_bytes(file[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(file[root..root + 2].try_into().unwrap()) as usize;
    let next = root + 2 + 12 * count;
    let start = file.len() as u32;
    file[next..next + 4].copy_from_slice(&start.to_le_bytes());
    let jpeg_at = start + 2 + 4 * 12 + 4;
    file.extend(4u16.to_le_bytes());
    for (tag, value) in [(256u16, w as u32), (257, h as u32), (0x201, jpeg_at), (0x202, jpeg.len() as u32)] {
        file.extend(tag.to_le_bytes());
        file.extend(4u16.to_le_bytes());
        file.extend(1u32.to_le_bytes());
        file.extend(value.to_le_bytes());
    }
    file.extend(0u32.to_le_bytes());
    file.extend(&jpeg);
    file
}

fn camera_tone_with(bytes: &[u8], camera: &str, look: bool) -> Option<lcc_pipeline::tone::CameraTone> {
    let options = LoadOptions { camera: camera.into(), camera_look: look, ..Default::default() };
    crate::files::load_bytes_with(bytes, 96, options).unwrap().1.camera_tone
}

/// A camera profile without a tone curve of its own (a chart-made matrix; also one with a look table, as
/// OneSlips decides) keeps the tone the photo has without it: here the tone fitted to the file's camera JPEG, not
/// the flat default tone. A profile with a tone curve of its own still renders with that curve.
#[test]
fn a_camera_profile_without_a_tone_curve_keeps_the_photos_own_tone() {
    let _lock = install();
    let bytes = dng_with_camera_jpeg();
    let own = camera_tone_with(&bytes, "lc", true).expect("without a profile: the tone fitted to the camera JPEG");
    let camera = 1.0 - (-3.0f32 * 0.18).exp();
    assert!((own.apply(0.18) - camera).abs() < 0.05, "the fit learnt the camera's curve: {} vs {camera}", own.apply(0.18));
    let default = lcc_pipeline::tone::CameraTone::default_raw().unwrap();
    assert!((own.apply(0.18) - default.apply(0.18)).abs() > 0.02, "the camera's tone is not the default one");
    // (kept under a profile without a curve, the tone is marked for Lightroom's default look in process 9)
    assert!(!own.has_default_curve(), "fitted without a profile: no Lightroom look");
    let kept = own.under_default_curve();
    assert_eq!(camera_tone_with(&bytes, PLAIN, true), Some(kept), "a matrix-only profile keeps the photo's tone");
    assert_eq!(camera_tone_with(&bytes, TABLE, true), Some(kept), "a look table is colour, not tone");
    assert_eq!(camera_tone_with(&bytes, TABLE, false), Some(kept), "under another look: the calibration and the photo's tone");
    // a profile with its own curve keeps it (As profile), and under another look its calibration only
    let curve = profile("x", true).color.profile.tone_curve.unwrap();
    let with_look = camera_tone_with(&bytes, WITH_LOOK, true).expect("the profile's own curve");
    assert_eq!(Some(with_look), crate::files::dng_tone_curve(&curve));
    assert!(!with_look.has_default_curve(), "a profile's own curve gets no Lightroom look");
    assert_eq!(camera_tone_with(&bytes, WITH_LOOK, false), None);
    // the colour still comes from the profile
    let options = LoadOptions { camera: PLAIN.into(), ..Default::default() };
    let colour = crate::files::load_bytes_with(&bytes, 96, options).unwrap().1.camera_color.map(|c| c.tags.color_matrix);
    assert_eq!(colour, Some(profile("x", false).color.color_matrix));
    // `photo.sourceReport` names the kept tone
    let report = crate::source_report::report(&bytes, PLAIN).unwrap();
    assert_eq!((report["tone"].as_str(), report["keptTone"].as_str()), (Some("withoutProfile"), Some("dngToneFit")), "{report}");
    // the render's contrast (luma standard deviation, 8-bit): the photo's tone against the default tone the profile
    // got before
    let options = LoadOptions { camera: PLAIN.into(), ..Default::default() };
    let (img, info) = crate::files::load_bytes_with(&bytes, 384, options).unwrap();
    // (mean, standard deviation) of the render's luma at a process version
    let luma = |info: &lcc_pipeline::SourceInfo, process: lcc_develop::Process| {
        let mut s = lcc_develop::DevelopSettings::for_raw(info.as_shot_temp, info.as_shot_tint);
        s.process = process.version();
        let out = lcc_pipeline::render(&img, info, &s, &lcc_pipeline::RenderRequest::fit(384, 256)).image;
        let luma: Vec<f64> = out.data.iter().map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])).collect();
        let mean = luma.iter().sum::<f64>() / luma.len() as f64;
        (mean, (luma.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / luma.len() as f64).sqrt())
    };
    let (_, kept) = luma(&info, lcc_develop::Process::V7);
    let (_, flat) = luma(&lcc_pipeline::SourceInfo { camera_tone: None, ..info.clone() }, lcc_develop::Process::V7);
    eprintln!("luma standard deviation under the matrix-only profile: {kept:.2} (default tone before: {flat:.2})");
    assert!(kept > flat * 1.05, "the photo's tone keeps more contrast: {kept} vs {flat}");
    // process 9 draws the kept tone in Lightroom's default look: lighter than process 7
    let (v7, v9) = (luma(&info, lcc_develop::Process::V7).0, luma(&info, lcc_develop::Process::V9).0);
    assert!(v9 > v7 + 2.0, "process 9 renders the kept tone lighter: {v9} vs {v7}");
}

/// An imported photo carries its camera profile in its settings (`develop.profile.camera`, from the import's
/// camera-profile choice); its grid and loupe renders decode with the photo's own tone under a matrix-only profile.
#[test]
fn an_imported_photo_with_a_matrix_only_profile_renders_with_the_photos_tone() {
    let _lock = install();
    let bytes = dng_with_camera_jpeg();
    let own = camera_tone_with(&bytes, "lc", true).expect("the tone fitted to the camera JPEG");
    let dir = std::env::temp_dir().join(format!("lc-camera-layers-import-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("CHART0001.DNG");
    std::fs::write(&path, &bytes).unwrap();
    let mut s = Session::new().with_fs();
    let r = s.execute("library.import", &json!({"paths": [path.to_string_lossy()], "cameraProfile": {"all": PLAIN}})).unwrap();
    let id = PhotoId(r["imported"][0].as_u64().unwrap());
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!(p.develop.profile.camera, PLAIN, "the profile is in the photo's settings");
    assert_eq!(LoadOptions::for_photo(&p).camera, PLAIN);
    for (what, job) in [("grid", s.thumb_job(id, 128)), ("loupe", s.render_job(id, 400, 300, false, true))] {
        let source = job.unwrap().source.load_source().unwrap();
        let tone = source.info.as_ref().and_then(|i| i.camera_tone).or(source.camera_tone);
        assert_eq!(tone, Some(own.under_default_curve()), "{what}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A matrix profile with a hue/saturation map that leaves greys alone and saturates every colour 1.8 times.
fn saturation_map_profile() -> lcc_raw::dcp::Dcp {
    let mut d = profile("Saturation Map", false);
    let mut data: Vec<f64> = Vec::new();
    for _ in 0..6 {
        for si in 0..3 {
            data.extend_from_slice(&[0.0, if si == 0 { 1.0 } else { 1.8 }, 1.0]);
        }
    }
    d.color.profile.hue_sat_map = [lcc_raw::profile::HsvTable::from_tags(&[6, 3, 1], &data, 0), None];
    assert!(d.color.profile.hue_sat_map[0].is_some());
    d
}

/// Process V9 decodes a raw without its camera profile's tables and hands them to the pipeline, which applies them
/// after the white balance; V1–V7 photos decode as they always did (the tables at the as-shot white). At the as-shot
/// white both render alike; away from it they differ (the V9 order).
#[test]
fn process_v9_hands_the_profile_tables_to_the_white_balance() {
    use lcc_develop::{DevelopSettings, Process, WbMode, WbScale};
    let _lock = install();
    let bytes = dng();
    let decode = |defer: bool| {
        let options = LoadOptions { camera: SATMAP.into(), defer_tables: defer, ..Default::default() };
        crate::files::load_bytes_with(&bytes, 64, options).unwrap()
    };
    let ((img7, info7), (img9, info9)) = (decode(false), decode(true));
    let cc7 = info7.camera_color.clone().expect("the camera's own colour model");
    let cc9 = info9.camera_color.clone().expect("the camera's own colour model");
    assert_eq!(cc7.deferred_tables_gain, None, "V1-V7: the tables are in the pixels");
    assert!(cc7.tags.profile.hue_sat_map.iter().all(Option::is_none));
    assert!(cc9.deferred_tables_gain.is_some() && cc9.tags.profile.hue_sat_map[0].is_some(), "V9: the tables wait for the white balance");
    assert!(img7.data != img9.data, "the V9 source leaves the tables out");
    // the settings choose the decode
    let mut v7 = DevelopSettings::for_raw(info7.as_shot_temp, info7.as_shot_tint);
    v7.process = Process::V7.version();
    let v9 = DevelopSettings { process: Process::V9.version(), ..v7.clone() };
    assert!(!LoadOptions::of(&v7).defer_tables && LoadOptions::of(&v9).defer_tables);
    // a decode without a photo's settings is the default settings' (the latest process): its cached source serves them
    // (a plain decode keeps the tables in the pixels, for callers that use them without the pipeline)
    assert!(!LoadOptions::default().defer_tables);
    let req = lcc_pipeline::RenderRequest::fit(64, 48);
    let max_diff = |a: &lcc_raster::Rgba8, b: &lcc_raster::Rgba8| {
        a.data.iter().zip(&b.data).flat_map(|(p, q)| (0..3).map(move |k| (i32::from(p[k]) - i32::from(q[k])).abs())).max().unwrap_or(0)
    };
    let at_shot = max_diff(&lcc_pipeline::render(&img7, &info7, &v7, &req).image, &lcc_pipeline::render(&img9, &info9, &v9, &req).image);
    assert!(at_shot <= 2, "at the as-shot white V9 renders as V7 did: {at_shot} levels apart");
    let warmer = |s: &DevelopSettings| {
        let mut s = s.clone();
        (s.wb.mode, s.wb.temp, s.wb.tint, s.wb.scale) = (WbMode::Custom, (info7.as_shot_temp * 0.6).max(2000.0), info7.as_shot_tint, WbScale::Kelvin);
        s
    };
    let moved =
        max_diff(&lcc_pipeline::render(&img7, &info7, &warmer(&v7), &req).image, &lcc_pipeline::render(&img9, &info9, &warmer(&v9), &req).image);
    assert!(moved >= 3, "away from it the tables follow the chosen white: {moved} levels apart");
}

/// Process V9 on a real calibrated raw (the public Canon EOS R6 sample, skipped without the corpus). Decoded at a size
/// the sensor bins to exactly (684 px: nothing resamples it after the colour model), the scene-linear picture after
/// V9's white balance is V7's decode at the as-shot white, pixel for pixel; away from it the two differ. (Where a
/// render resamples the source before the white balance, V9 applies the tables to the resampled pixels and V7 had
/// them in the pixels before: on this calibration's look table that moves a 1600 px preview by about 1 dE on average.)
#[test]
fn corpus_v9_at_the_as_shot_white_develops_a_calibrated_raw_as_v7() {
    use lcc_develop::{DevelopSettings, Process, WbMode, WbScale};
    let path = lcc_brand::env_os("CORPUS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus"))
        .join("raw/cr3-canon-r6-craw.cr3");
    let Ok(bytes) = std::fs::read(&path) else {
        lcc_testkit::skip!(Corpus, "{} absent", path.display());
        return;
    };
    crate::camera_profiles::install_dcps();
    let decode = |defer: bool| crate::files::load_bytes_with(&bytes, 684, LoadOptions { defer_tables: defer, ..Default::default() }).unwrap();
    let ((img7, info7), (img9, info9)) = (decode(false), decode(true));
    assert_eq!(img7.width.max(img7.height), 684, "binned to the size asked for, not resampled");
    assert!(info9.camera_color.as_ref().is_some_and(|c| c.deferred_tables_gain.is_some()), "the R6 calibration's tables wait");
    let mut v7 = DevelopSettings::for_raw(info7.as_shot_temp, info7.as_shot_tint);
    v7.process = Process::V7.version();
    let v9 = DevelopSettings { process: Process::V9.version(), ..v7.clone() };
    let developed = |img: &lcc_raster::Rgb32f, info: &lcc_pipeline::SourceInfo, s: &DevelopSettings| {
        let mut img = img.clone();
        lcc_pipeline::local::white_balance(&mut img, info, s);
        img
    };
    let worst = |a: &lcc_raster::Rgb32f, b: &lcc_raster::Rgb32f| {
        a.data.iter().zip(&b.data).map(|(p, q)| (0..3).map(|k| (p[k] - q[k]).abs() / p[k].abs().max(0.01)).fold(0f32, f32::max)).fold(0f32, f32::max)
    };
    let at_shot = worst(&developed(&img7, &info7, &v7), &developed(&img9, &info9, &v9));
    assert!(at_shot < 1e-4, "at the as-shot white the pictures differ by {at_shot} (relative)");
    let warm = |s: &DevelopSettings| {
        let mut s = s.clone();
        (s.wb.mode, s.wb.temp, s.wb.tint, s.wb.scale) = (WbMode::Custom, 3200.0, 0.0, WbScale::Kelvin);
        s
    };
    let moved = worst(&developed(&img7, &info7, &warm(&v7)), &developed(&img9, &info9, &warm(&v9)));
    assert!(moved > 0.02, "at 3200 K the tables follow the chosen white: {moved}");
}
