//! Culling a browsed folder (raws not imported yet) in the loupe: every step shows a picture at
//! once. A browsed raw renders with its body's own profile and never with the Import panel's
//! choice ([`lcc_engine::Session::browse_profile`], as in OneSlips), and the view asks for the
//! look it renders with ([`lcc_engine::Session::placement_of`]), so its camera JPEG, built preview
//! and earlier render stand in for it. Before (2026-11-23), the Import panel's profile reached the
//! render: every picture of the photo had a look other than the record's, the loupe took none of
//! them, and each step showed "Rendering…" on an empty canvas until the photo's own render came
//! back, forward and back alike.
//!
//! The user's rules, one test each, with a camera profile picked in the Import panel:
//! 1. back to a photo already visited: its picture shows at once, with no render;
//! 2. once the preview build is done: every step shows a finished picture at once;
//! 3. before that: the cheapest picture (the camera JPEG) at once, never an empty canvas.
//!
//! Each also for a body with a default profile of its own (set in the app).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use lcc_catalog::{Op, Photo, PhotoId, Source};
use serde_json::json;

use crate::headless::Headless;
use crate::render::Slot;
use crate::{LccApp, Services};

const T: Duration = Duration::from_secs(30);

/// How long a raw decode takes once the test has set things up: far longer than a step, so a
/// picture that shows within [`AT_ONCE`] was not rendered for the step.
const SLOW_MS: u64 = 4000;

/// What "at once" allows: a few frames of the headless loop (each 5 ms plus the frame) and the
/// stand-in job reading a picture that is in memory or on disk.
const AT_ONCE: Duration = Duration::from_millis(800);

/// How the browsed raws get a camera profile other than the camera's default.
#[derive(Clone, Copy, Debug)]
enum Profile {
    /// One picked for all raws in the Import panel (for the import; the browse view ignores it).
    ImportPanel,
    /// Their body's default, set in the app.
    BodyDefault,
}

/// A headless app with `n` browsed raws (not imported) of one Canon body with a camera profile as
/// `profile` says: their camera JPEGs read at once, a raw decode taking `delay` ms.
fn browsed(n: u64, delay: Arc<AtomicU64>, profile: Profile) -> Headless {
    browsed_of(n, delay, profile, (600, 400))
}

/// [`browsed`] with raws that decode to `size` pixels.
fn browsed_of(n: u64, delay: Arc<AtomicU64>, profile: Profile, size: (usize, usize)) -> Headless {
    let mut s = lcc_engine::Session::new();
    s.media.preview_loader = Some(Arc::new(|_: &str, edge: usize| {
        let w = edge.clamp(1, 300);
        let h = (w * 2 / 3).max(1);
        Some(lcc_raster::Rgba8 { width: w, height: h, data: vec![[200, 60, 200, 255]; w * h] })
    }));
    s.media.file_loader = Some(Arc::new(move |_, _, _| {
        std::thread::sleep(Duration::from_millis(delay.load(Ordering::Relaxed)));
        let mut img = lcc_raster::Rgb32f::new(size.0, size.1);
        img.data.fill([0.18, 0.18, 0.18]);
        Ok((img, Default::default()))
    }));
    for id in 1..=n {
        let mut p =
            Photo::new(PhotoId(id), Source::File { path: format!("shoot/b{id}.cr3") }, &format!("b{id}.cr3"), "CR3", 6000, 4000, "2026-11-19");
        p.kind = lcc_catalog::MediaKind::Raw;
        p.meta.camera = "Canon EOS R5 C".into();
        p.meta.camera_model = "Canon EOS R5 C".into();
        p.meta.camera_serial = "653499300171".into();
        p.develop = Arc::new(p.camera_defaults());
        p.local = true;
        p.set_local_baseline();
        s.catalog.apply(Op::AddPhoto { photo: Box::new(p) }).unwrap();
    }
    match profile {
        Profile::ImportPanel => s.import_profile_choice = lcc_engine::profile_bodies::ImportProfile::All("lc".into()),
        Profile::BodyDefault => {
            s.body_profiles.insert("Canon EOS R5 C|653499300171".into(), "lc".into());
        }
    }
    let app = LccApp::new(s, Services { png: None, ..Default::default() });
    Headless::new(app, [1400.0, 900.0], 1.0)
}

fn exec(h: &mut Headless, command: &str, params: serde_json::Value) {
    let r = h.request("engine.execute", json!({"command": command, "params": params}), T);
    assert_eq!(r["ok"], true, "{command}: {r}");
}

/// Step frames until `done` (at most `max`), handing every frame to `each`.
fn frames(h: &mut Headless, max: Duration, mut each: impl FnMut(&Headless), done: impl Fn(&Headless) -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < max {
        h.step();
        each(h);
        if done(h) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// What the loupe shows of photo `id` this frame (`None`: another photo, or nothing yet).
fn shown(h: &Headless, id: u64) -> Option<&'static str> {
    h.app.loupe_shown.filter(|(p, _)| *p == PhotoId(id)).map(|(_, w)| w)
}

/// Show photo `id` in the loupe and wait for its own render.
fn visit(h: &mut Headless, id: u64) {
    exec(h, "library.select", json!({"ids": [id]}));
    let r = h.request("ui.set", json!({"view": "detail"}), T);
    assert_eq!(r["ok"], true, "{r}");
    let done = |h: &Headless| {
        shown(h, id) == Some("render")
            && !h.app.renderer.is_pending(Slot::Main)
            && h.app.renderer.textures.get(&Slot::Main).is_some_and(|t| t.photo == PhotoId(id))
    };
    assert!(frames(h, T, |_| {}, done), "photo {id} rendered: {:?}", h.app.loupe_shown);
}

/// Step to photo `id` and record what the loupe showed of it, frame by frame, for `max`: the time,
/// the source and whether the picture has the photo's look (not one standing in with another).
fn step_to(h: &mut Headless, id: u64, max: Duration) -> Vec<(Duration, &'static str, bool)> {
    exec(h, "library.select", json!({"ids": [id]}));
    let t0 = Instant::now();
    let mut seen = Vec::new();
    frames(
        h,
        max,
        |h| {
            let what = shown(h, id).unwrap_or("none");
            let own_look = what != "none" && h.app.loupe_drawn.is_some_and(|d| d.known);
            seen.push((t0.elapsed(), what, own_look));
        },
        |_| false,
    );
    seen
}

/// The first picture of the step with the photo's own look and a source `ok` takes shows within
/// [`AT_ONCE`], and every frame after it shows a picture with that look.
fn at_once(seen: &[(Duration, &'static str, bool)], ok: impl Fn(&str) -> bool) -> bool {
    seen.iter().position(|(_, w, own)| *own && ok(w)).is_some_and(|i| seen[i].0 <= AT_ONCE && seen[i..].iter().all(|(_, _, own)| *own))
}

/// Rule 3: a browsed raw never seen in the loupe shows its camera JPEG at once, whatever profile
/// the Import panel picked, and the canvas is never empty until its render replaces it.
#[test]
fn a_browsed_raw_shows_its_camera_jpeg_at_once() {
    let failed: Vec<_> =
        [Profile::ImportPanel, Profile::BodyDefault].into_iter().filter(|p| std::panic::catch_unwind(|| camera_jpeg_at_once(*p)).is_err()).collect();
    assert!(failed.is_empty(), "failed for {failed:?}");
}

fn camera_jpeg_at_once(profile: Profile) {
    let delay = Arc::new(AtomicU64::new(0));
    let mut h = browsed(3, delay.clone(), profile);
    visit(&mut h, 1);
    delay.store(SLOW_MS, Ordering::Relaxed);
    let seen = step_to(&mut h, 3, Duration::from_millis(1500));
    assert!(at_once(&seen, |w| w == "embedded"), "{profile:?}: the camera JPEG at once, then always a picture: {seen:?}");
    delay.store(0, Ordering::Relaxed);
}

/// Rule 1: back to photos already visited, each shows its own earlier render (kept as its view)
/// at once: nothing renders for the step.
#[test]
fn going_back_to_a_visited_browsed_photo_shows_its_render_at_once() {
    let failed: Vec<_> =
        [Profile::ImportPanel, Profile::BodyDefault].into_iter().filter(|p| std::panic::catch_unwind(|| back_at_once(*p)).is_err()).collect();
    assert!(failed.is_empty(), "failed for {failed:?}");
}

fn back_at_once(profile: Profile) {
    let delay = Arc::new(AtomicU64::new(0));
    let mut h = browsed(8, delay.clone(), profile);
    for id in 1..=8 {
        visit(&mut h, id);
    }
    // (slow from here: what shows at once was kept, or made, before the step)
    delay.store(SLOW_MS, Ordering::Relaxed);
    for id in [2, 1, 3] {
        let seen = step_to(&mut h, id, Duration::from_millis(1000));
        // (a render on screen within AT_ONCE of the step was made before it: a decode takes SLOW_MS)
        assert!(at_once(&seen, |w| matches!(w, "cached" | "last" | "render")), "{profile:?}, photo {id}: its earlier render at once: {seen:?}");
    }
    delay.store(0, Ordering::Relaxed);
}

/// Rule 2: once the previews are built, every photo shows its finished picture at once on its
/// first step, the camera profile's look included.
#[test]
fn after_the_preview_build_every_browsed_photo_shows_its_preview_at_once() {
    let failed: Vec<_> =
        [Profile::ImportPanel, Profile::BodyDefault].into_iter().filter(|p| std::panic::catch_unwind(|| built_at_once(*p)).is_err()).collect();
    assert!(failed.is_empty(), "failed for {failed:?}");
}

fn built_at_once(profile: Profile) {
    let delay = Arc::new(AtomicU64::new(0));
    let mut h = browsed(5, delay.clone(), profile);
    visit(&mut h, 1);
    let r = h.app.session.execute("library.buildPreviews", &json!({"size": "standard", "ids": [2, 3, 4, 5], "wait": true}));
    assert!(r.is_ok(), "{r:?}");
    delay.store(SLOW_MS, Ordering::Relaxed);
    for id in 2..=5 {
        let seen = step_to(&mut h, id, Duration::from_millis(1000));
        // (the stand-in, or the loupe's own picture taken from the built preview: on screen within
        // AT_ONCE of the step, it was not decoded for it, a decode takes SLOW_MS)
        assert!(at_once(&seen, |w| matches!(w, "cached" | "render")), "{profile:?}, photo {id}: its built preview at once: {seen:?}");
    }
    delay.store(0, Ordering::Relaxed);
}

/// Use a wide-gamut monitor profile, as Settings ▸ Display does (the user's monitor has one of its own).
fn use_monitor_profile(h: &mut Headless) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lcc-cull-monitor-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("monitor.icc");
    std::fs::write(&path, lcc_codecs::icc::write_named(lcc_codecs::NamedSpace::DisplayP3)).unwrap();
    h.app.ui.settings.display_profile = path.to_string_lossy().into_owned();
    h.step();
    assert!(h.app.renderer.display().is_some(), "the monitor profile is in use");
    dir
}

/// Half a raw decode ([`SLOW_MS`]): the loupe's own picture of a photo on screen this soon after the
/// step was not decoded for it. (Longer than [`AT_ONCE`]: the neighbours' and the histogram's jobs
/// share the workers with it, and tests run side by side.)
const NO_RENDER: Duration = Duration::from_millis(SLOW_MS / 2);

/// Step to photo `id`: whether the loupe's own picture of it (not a stand-in) is on screen within
/// [`NO_RENDER`], with no render of it still out.
fn loupe_has_it_at_once(h: &mut Headless, id: u64) -> bool {
    exec(h, "library.select", json!({"ids": [id]}));
    frames(
        h,
        NO_RENDER,
        |_| {},
        |h| !h.app.renderer.is_pending(Slot::Main) && h.app.renderer.textures.get(&Slot::Main).is_some_and(|t| t.photo == PhotoId(id)),
    )
}

/// Rules 2 and 1 on a monitor with its own colour profile: once the previews are built, and going
/// back over photos already seen, the loupe's picture of each photo is the one it has (a built
/// preview, a render kept from the visit), made the monitor's as every cached picture is; nothing
/// is decoded or rendered for the step. Before (2026-11-23), a profiled monitor made the loupe
/// refuse every kept picture: behind the stand-in, each step decoded and rendered the raw again
/// (1-4 s of the machine a photo on the user's set-up), forward after the build and back.
#[test]
fn on_a_profiled_monitor_the_loupe_renders_nothing_it_already_has() {
    let delay = Arc::new(AtomicU64::new(0));
    // (raws bigger than the loupe draws them: a built preview serves it only when it is as big)
    let mut h = browsed_of(6, delay.clone(), Profile::BodyDefault, (1800, 1200));
    let profile_dir = use_monitor_profile(&mut h);
    visit(&mut h, 1);
    let r = h.app.session.execute("library.buildPreviews", &json!({"size": "standard", "ids": [2, 3, 4, 5, 6], "wait": true}));
    assert!(r.is_ok(), "{r:?}");
    delay.store(SLOW_MS, Ordering::Relaxed);
    for id in [2, 3, 4, 5, 6, 5, 4, 3, 2, 1] {
        assert!(
            loupe_has_it_at_once(&mut h, id),
            "photo {id}: the loupe's picture at once, nothing rendered: {:?}, waiting for {:?}, {} running",
            h.app.loupe_shown,
            h.app.renderer.pending_slots(),
            h.app.renderer.in_flight()
        );
    }
    delay.store(0, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(profile_dir);
}

/// While the loupe shows only a photo's thumbnail and its full-size camera JPEG is on its way,
/// background work waits for it (OneSlips PERF-CULL-TIERS: the photo on screen comes first, at its
/// cheapest first). Before (2026-11-23), only an empty loupe held it back: the first photos after
/// the loupe opened decoded their camera JPEG beside the background build's raw decodes, 0.13-1.8 s
/// instead of 28 ms, and a step went without its full-size picture.
#[test]
fn a_thumbnail_on_screen_holds_background_work_until_the_camera_jpeg_is_in() {
    let delay = Arc::new(AtomicU64::new(SLOW_MS));
    // (the raw decodes slowly: no render overtakes the stand-ins)
    let mut h = browsed(3, delay, Profile::BodyDefault);
    // the full-size camera JPEG takes a while; the small one a thumbnail is made from comes at once
    h.app.session.media.preview_loader = Some(Arc::new(|_: &str, edge: usize| {
        if edge > 512 {
            std::thread::sleep(Duration::from_millis(1500));
        }
        let w = edge.clamp(1, 2000);
        let h = (w * 2 / 3).max(1);
        Some(lcc_raster::Rgba8 { width: w, height: h, data: vec![[200, 60, 200, 255]; w * h] })
    }));
    // its thumbnail first (the Import tab's grid and the filmstrip make them before the loupe opens)
    exec(&mut h, "library.select", json!({"ids": [1]}));
    let t0 = Instant::now();
    while h.app.renderer.thumb(PhotoId(1)).is_none() && t0.elapsed() < T {
        crate::panels::grid::request_thumb(&mut h.app, PhotoId(1), 256, 97);
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(h.app.renderer.thumb(PhotoId(1)).is_some(), "the thumbnail is made");
    let r = h.request("ui.set", json!({"view": "detail"}), T);
    assert_eq!(r["ok"], true, "{r}");
    // the thumbnail up in the loupe, its camera JPEG still coming
    let thumb_up = |h: &Headless| shown(h, 1) == Some("thumb") && h.app.renderer.is_pending(Slot::Preview);
    assert!(frames(&mut h, T, |_| {}, thumb_up), "the thumbnail first: {:?}", h.app.loupe_shown);
    // (whatever held it back so far: from here, the loupe alone)
    h.app.pictures_missing_pass = None;
    let mut held = false;
    frames(&mut h, Duration::from_millis(300), |h| held |= thumb_up(h) && h.app.pictures_missing_pass.is_some(), |_| false);
    assert!(held, "background work waits while only the thumbnail is on screen");
}
