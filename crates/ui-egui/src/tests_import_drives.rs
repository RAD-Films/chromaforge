//! Import ▸ Import from lists This PC: every drive and volume as a folder tree (tests see one fake
//! drive, the temporary folder, so they never touch this machine's drives), above the usual places.

use std::time::Duration;

use serde_json::json;

use crate::LccApp;
use crate::headless::Headless;
use crate::state::ViewMode;

const T: Duration = Duration::from_secs(60);

/// Both tests use the one test drive (a folder in the temporary folder, removed at the end of each).
static DRIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn import_from_lists_the_drives_as_folder_trees() {
    let _drive = DRIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut app = LccApp::new(lcc_engine::Session::new(), crate::Services { png: None, ..Default::default() });
    app.ui.view = ViewMode::PhotoGrid;
    let mut h = Headless::new(app, [1400.0, 900.0], 1.0);
    h.settle(T);
    let r = h.request("engine.execute", json!({"command": "dialog.import", "params": {}}), T);
    assert_eq!(r["ok"], true, "{r}");
    let root = crate::import_tab::test_drive();
    std::fs::create_dir_all(root.join("DCIM")).unwrap();
    let drive = root.to_string_lossy().to_string();
    let row = format!("source:local:{drive}");
    // the drives are read on a worker thread: they show a few frames later
    h.step_until(Duration::from_secs(10), |h| h.app.widgets.iter().any(|(w, _)| *w == row));
    assert!(h.app.widgets.iter().any(|(w, _)| *w == row), "the drive's row in Import from");
    assert!(h.app.widgets.iter().any(|(w, _)| *w == format!("folderToggle:{drive}")), "it opens as a tree");
    // a click browses it
    assert_eq!(h.request("ui.clickWidget", json!({"id": row}), T)["ok"], true);
    h.settle(T);
    let browsed = h.app.session.browse.as_ref().map(|b| std::path::PathBuf::from(&b.path));
    assert_eq!(browsed.as_deref(), Some(root.as_path()));
    let _ = std::fs::remove_dir_all(&root);
}

/// Issue #1234, deeper in the tree: the arrows of folders inside a drive (a shoot, its day) only open
/// and close their subfolder lists. None of them changes the source or starts reading a folder; the
/// folder's name then loads its photos.
#[test]
fn import_subfolder_arrows_only_expand_and_start_no_scan() {
    let _drive = DRIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut app = LccApp::new(lcc_engine::Session::new().with_fs(), crate::Services { png: None, ..Default::default() });
    app.ui.view = ViewMode::PhotoGrid;
    let mut h = Headless::new(app, [1400.0, 900.0], 1.0);
    h.settle(T);
    let r = h.request("engine.execute", json!({"command": "dialog.import", "params": {}}), T);
    assert_eq!(r["ok"], true, "{r}");
    let root = crate::import_tab::test_drive();
    let (shoot, day) = (root.join("Shoot"), root.join("Shoot").join("Day 1"));
    std::fs::create_dir_all(day.join("Card")).unwrap();
    let o = lcc_engine::export::ExportOptions { format: lcc_engine::export::ExportFormat::Png, ..Default::default() };
    let img = lcc_raster::Rgba8::from_fn(24, 16, |x, y| [(x * 9) as u8, (y * 12) as u8, 77, 255]);
    std::fs::write(day.join("p.png"), lcc_engine::export::encode_image(&img, &o).unwrap()).unwrap();
    let path = |p: &std::path::Path| p.to_string_lossy().to_string();
    let rect = |h: &Headless, id: &str| h.app.widgets.iter().find(|(w, _)| w == id).map(|(_, r)| *r);
    let source_before = h.app.session.source;
    let untouched = |h: &Headless, what: &str| {
        assert_eq!(h.app.session.browse, None, "{what} must not load a folder");
        assert!(h.app.scan.is_none() && h.app.import.is_none(), "{what} must not start reading a folder");
        assert_eq!(h.app.session.source, source_before, "{what} must not change the source");
    };
    // the drive's arrow, then the shoot's, then the day's: each lists the next level down
    for (arrow, next) in [(path(&root), path(&shoot)), (path(&shoot), path(&day)), (path(&day), path(&day.join("Card")))] {
        let arrow = format!("folderToggle:{arrow}");
        h.step_until(Duration::from_secs(10), |h| rect(h, &arrow).is_some());
        let Some(at) = rect(&h, &arrow) else { panic!("{arrow}") };
        press_release_at(&mut h, at.center());
        let next = format!("source:local:{next}");
        h.step_until(Duration::from_secs(10), |h| rect(h, &next).is_some());
        assert!(rect(&h, &next).is_some(), "{arrow} lists {next}");
        untouched(&h, &arrow);
    }
    // the day's arrow closes it again, still loading nothing
    let card = format!("source:local:{}", path(&day.join("Card")));
    let Some(at) = rect(&h, &format!("folderToggle:{}", path(&day))) else { panic!("the day's arrow") };
    press_release_at(&mut h, at.center());
    h.step_until(Duration::from_secs(10), |h| rect(h, &card).is_none());
    assert!(rect(&h, &card).is_none(), "the day's arrow closes it");
    untouched(&h, "closing the day's arrow");
    // the day's name loads it: its photo is listed
    let Some(name) = rect(&h, &format!("source:local:{}", path(&day))) else { panic!("the day's row") };
    press_release_at(&mut h, name.center());
    h.step_until(Duration::from_secs(10), |h| h.app.session.browse.is_some() && h.app.scan.is_none() && h.app.import.is_none());
    h.settle(T);
    let browsed = h.app.session.browse.as_ref().map(|b| std::path::PathBuf::from(&b.path));
    assert_eq!(browsed.as_deref(), Some(day.as_path()), "the name browses the folder");
    assert_eq!(h.app.session.source, lcc_engine::LibrarySource::Folder);
    assert_eq!(h.app.session.visible_cloned().len(), 1, "its photo is listed");
    let _ = std::fs::remove_dir_all(&root);
}

/// Press, a frame, release at `at` with the pointer moved there first: how a mouse (or a finger) comes in.
fn press_release_at(h: &mut Headless, at: egui::Pos2) {
    let primary = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    h.events.push(egui::Event::PointerMoved(at));
    h.step();
    h.events.push(primary(true));
    h.step();
    h.events.push(primary(false));
    h.settle(T);
}

/// The arrow of a folder in Import from only opens or closes its subfolder list; the name browses it
/// (issue #1234). A card is listed as a drive of its own and also lies inside the disk it is mounted
/// on (`/` ▸ `/run/media/me/CARD`): its two rows used to share one arrow id, so the copy drawn first
/// had no arrow of its own and a click on it fell through to its row and browsed the folder.
#[test]
fn import_folder_arrow_only_expands_and_the_name_loads() {
    let _drive = DRIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut app = LccApp::new(lcc_engine::Session::new(), crate::Services { png: None, ..Default::default() });
    app.ui.view = ViewMode::PhotoGrid;
    let mut h = Headless::new(app, [1400.0, 900.0], 1.0);
    h.settle(T);
    let r = h.request("engine.execute", json!({"command": "dialog.import", "params": {}}), T);
    assert_eq!(r["ok"], true, "{r}");
    let root = crate::import_tab::test_drive();
    let card = crate::import_tab::test_card();
    std::fs::create_dir_all(card.join("DCIM")).unwrap();
    let (drive, card) = (root.to_string_lossy().to_string(), card.to_string_lossy().to_string());
    let rect = |h: &Headless, id: &str| h.app.widgets.iter().find(|(w, _)| w == id).map(|(_, r)| *r);
    let browsed = |h: &Headless| h.app.session.browse.as_ref().map(|b| b.path.clone());
    let (drive_arrow, card_row, card_arrow) = (format!("folderToggle:{drive}"), format!("source:local:{card}"), format!("folderToggle:{card}"));
    h.step_until(Duration::from_secs(10), |h| rect(h, &drive_arrow).is_some());
    // the disk's arrow opens it: the card is listed inside it, above the card's own row
    let Some(arrow) = rect(&h, &drive_arrow) else { panic!("the disk's arrow") };
    press_release_at(&mut h, arrow.center());
    h.step_until(Duration::from_secs(10), |h| h.app.widgets.iter().filter(|(w, _)| *w == card_arrow).count() == 2);
    assert_eq!(browsed(&h), None, "the arrow must not load the folder");
    // the name loads it (the folder is read on a worker thread)
    let Some(name) = rect(&h, &card_row) else { panic!("the card's row") };
    press_release_at(&mut h, name.center());
    h.step_until(Duration::from_secs(10), |h| h.app.session.browse.is_some());
    let loaded = browsed(&h);
    assert_eq!(loaded.as_deref().map(std::path::Path::new), Some(root.join("Card").as_path()), "the name browses the folder");
    // each copy's arrow is its own: neither loads anything, both open and close their row (the rows are looked
    // up again for every click: opening a row moves the ones below it)
    for copy in 0..2 {
        for _ in 0..2 {
            let arrows: Vec<egui::Rect> = h.app.widgets.iter().filter(|(w, _)| *w == card_arrow).map(|(_, r)| *r).collect();
            assert!(arrows.len() >= 2, "the card is listed inside the disk and as a drive");
            press_release_at(&mut h, arrows[copy].center());
            assert_eq!(browsed(&h), loaded, "the arrow of copy {copy} must not load the folder");
        }
    }
    // and the name still loads a folder after its arrows were used
    let Some(name) = rect(&h, &format!("source:local:{drive}")) else { panic!("the disk's row") };
    press_release_at(&mut h, name.center());
    h.step_until(Duration::from_secs(10), |h| browsed(h).as_deref().map(std::path::Path::new) == Some(root.as_path()));
    assert_eq!(browsed(&h).as_deref().map(std::path::Path::new), Some(root.as_path()), "the disk's name browses it");
    let _ = std::fs::remove_dir_all(&root);
}
