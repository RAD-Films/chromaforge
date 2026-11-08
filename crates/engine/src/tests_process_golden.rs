//! The golden set of processes V1 to V8: every photo of the CC0 raw corpus and a few procedural scenes, rendered
//! under each frozen process with the import defaults, a busy edit, a 16-bit ProPhoto export and an HDR edit (and a raw's
//! full-size, demosaiced decode with the import defaults), must
//! keep the exact pixels they had when the hashes were recorded (on main at Clyde's V8, before process V9 was
//! stacked on it). A newer process may change anything; an older one never changes a saved photo's look.
//!
//! The hashes are per platform (`tests/data/process-golden/<os>-<arch>.json`: the maths libraries round the last
//! bit differently), so a platform without a file skips. The corpus part runs when the corpus is there
//! (`<PREFIX>_CORPUS`, default `corpus/` at the workspace root); the scenes always run where there are hashes.
//! Record (only on the commit the set is frozen at): `<PREFIX>_PROCESS_GOLDEN_WRITE=1 cargo test -p lcc-engine
//! --lib process_golden -- --ignored --nocapture`, with the corpus present.

use crate::media::LoadOptions;
use lcc_develop::{DevelopSettings, Hdr, Process};
use lcc_pipeline::{OutputDepth, OutputSpace, RenderRequest, SourceInfo};
use lcc_raster::Rgb32f;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Long edge of every render (and of the raw decodes, which bin to it).
const EDGE: usize = 512;
/// A decode limit above every raw's size: the full-size, demosaiced decode of an export.
const FULL: usize = 1 << 20;

/// The frozen processes.
fn frozen() -> Vec<Process> {
    Process::ALL.into_iter().filter(|p| p.version().0 <= 8).collect()
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/process-golden").join(format!("{}-{}.json", std::env::consts::OS, std::env::consts::ARCH))
}

fn corpus_dir() -> PathBuf {
    lcc_brand::env_os("CORPUS").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")).join("raw")
}

/// The edits every picture is rendered with, as (name, settings, request).
fn variants(base: &DevelopSettings) -> Vec<(&'static str, DevelopSettings, RenderRequest)> {
    let fit = RenderRequest::fit(EDGE, EDGE);
    let mut edit = base.clone();
    (edit.light.exposure, edit.light.contrast, edit.light.highlights, edit.light.shadows) = (0.7, 20.0, -60.0, 55.0);
    (edit.light.whites, edit.light.blacks) = (25.0, -15.0);
    (edit.effects.clarity, edit.effects.texture, edit.effects.dehaze) = (20.0, 15.0, 10.0);
    (edit.color.vibrance, edit.color.saturation) = (25.0, -10.0);
    (edit.detail.nr_luminance, edit.detail.nr_color, edit.detail.sharpen_amount) = (35.0, 40.0, 50.0);
    (edit.grain.amount, edit.grain.size, edit.grain.seed) = (20.0, 30.0, 7);
    let deep = RenderRequest { depth: OutputDepth::U16, space: OutputSpace::ProPhoto, ..fit.clone() };
    let mut hdr = edit.clone();
    // every field set: a newer process may start new HDR edits at another headroom
    hdr.hdr = Hdr { enabled: true, max_ev: 3.0, ..hdr.hdr.clone() };
    let hdr_req = RenderRequest { depth: OutputDepth::F32Hdr, ..fit.clone() };
    vec![("default", base.clone(), fit.clone()), ("edit", edit.clone(), fit), ("edit16", edit, deep), ("hdr", hdr, hdr_req)]
}

/// Hash of a render: its 8-bit picture and, when there is one, its deep picture.
fn render_hash(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, req: &RenderRequest) -> String {
    let r = lcc_pipeline::render(src, info, s, req);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut h, &(r.image.width as u64).to_le_bytes());
    fnv(&mut h, &(r.image.height as u64).to_le_bytes());
    for p in &r.image.data {
        fnv(&mut h, p);
    }
    if let Some(d) = &r.deep {
        match &d.samples {
            lcc_pipeline::DeepSamples::U16(v) => v.iter().for_each(|x| fnv(&mut h, &x.to_le_bytes())),
            lcc_pipeline::DeepSamples::F32(v) => v.iter().for_each(|x| fnv(&mut h, &x.to_bits().to_le_bytes())),
        }
    }
    format!("{h:016x}")
}

/// Every render of the golden set this machine can make: key → hash ("error: …" for a raw that doesn't decode).
fn render_set(with_corpus: bool) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (i, scene) in lcc_scenes::demo_library().iter().take(4).enumerate() {
        let src = scene.render_fit_serial(EDGE);
        for p in frozen() {
            let base = DevelopSettings { process: p.version(), ..DevelopSettings::default() };
            for (name, s, req) in variants(&base) {
                out.insert(format!("scene{i}|V{}|{name}", p.version().0), render_hash(&src, &SourceInfo::default(), &s, &req));
            }
        }
    }
    if !with_corpus {
        return out;
    }
    crate::camera_profiles::install_dcps();
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir()).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
    files.sort();
    for path in files {
        let Some(file) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let mut decoded: Vec<(LoadOptions, Result<(Rgb32f, SourceInfo), String>)> = Vec::new();
        let mut full: Vec<(LoadOptions, Result<(Rgb32f, SourceInfo), String>)> = Vec::new();
        for p in frozen() {
            // the import defaults of a raw, then this process; the decode follows the settings as a photo's does
            let probe = DevelopSettings { process: p.version(), ..DevelopSettings::default() };
            let options = LoadOptions::of(&probe);
            if !decoded.iter().any(|(o, _)| *o == options) {
                let d = crate::files::load_bytes_with(&bytes, EDGE, options.clone());
                decoded.push((options.clone(), d));
            }
            let Some((_, d)) = decoded.iter().find(|(o, _)| *o == options) else { continue };
            let (src, info) = match d {
                Ok(d) => d,
                Err(e) => {
                    out.insert(format!("{file}|V{}|decode", p.version().0), format!("error: {e}"));
                    continue;
                }
            };
            let mut base = DevelopSettings::for_raw(info.as_shot_temp, info.as_shot_tint);
            base.process = p.version();
            for (name, s, req) in variants(&base) {
                out.insert(format!("{file}|V{}|{name}", p.version().0), render_hash(src, info, &s, &req));
            }
            // the full-size decode an export or a 1:1 preview makes (demosaiced, not binned), rendered at EDGE
            if !full.iter().any(|(o, _)| *o == options) {
                full.push((options.clone(), crate::files::load_bytes_with(&bytes, FULL, options.clone())));
            }
            let Some((_, Ok((src, info)))) = full.iter().find(|(o, _)| *o == options) else { continue };
            out.insert(format!("{file}|V{}|full", p.version().0), render_hash(src, info, &base, &RenderRequest::fit(EDGE, EDGE)));
        }
    }
    out
}

#[test]
#[ignore = "records the golden set: run on the commit it is frozen at"]
fn process_golden_record() {
    if lcc_brand::env_os("PROCESS_GOLDEN_WRITE").is_none() {
        return;
    }
    let corpus = corpus_dir().is_dir();
    assert!(corpus, "the corpus belongs in the golden set: {} absent", corpus_dir().display());
    let first = render_set(true);
    // a render that isn't the same twice can't be frozen
    let again = render_set(true);
    let unstable: Vec<&String> = first.keys().filter(|k| first.get(*k) != again.get(*k)).collect();
    assert!(unstable.is_empty(), "not deterministic: {unstable:?}");
    let path = golden_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_string_pretty(&first).unwrap() + "\n").unwrap();
    eprintln!("wrote {} renders to {}", first.len(), path.display());
}

#[test]
fn processes_v1_to_v8_render_the_golden_set_unchanged() {
    let Ok(text) = std::fs::read_to_string(golden_path()) else {
        lcc_testkit::skip!(Input, "no golden hashes for this platform ({})", golden_path().display());
        return;
    };
    let golden: BTreeMap<String, String> = serde_json::from_str(&text).unwrap();
    let corpus = corpus_dir().is_dir();
    if !corpus {
        lcc_testkit::skip!(Corpus, "{} absent: only the procedural scenes are checked", corpus_dir().display());
    }
    let now = render_set(corpus);
    let mut changed = Vec::new();
    let mut checked = 0usize;
    for (key, want) in &golden {
        // a raw that didn't decode when the set was frozen may decode now (a reader fix is not a look change)
        if want.starts_with("error:") || (!corpus && !key.starts_with("scene")) {
            continue;
        }
        checked += 1;
        match now.get(key) {
            Some(got) if got == want => {}
            got => changed.push(format!("{key}: {want} → {}", got.map_or("missing", |g| g.as_str()))),
        }
    }
    eprintln!("golden set: {checked} renders checked, {} changed", changed.len());
    assert!(changed.is_empty(), "{} of {checked} frozen renders changed:\n{}", changed.len(), changed.join("\n"));
}
