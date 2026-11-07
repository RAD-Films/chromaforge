//! M11.4 end to end: `calibrate-lightroom add` refuses damaged or foreign files with an error exit (never a panic)
//! and leaves the repository untouched; bad arguments are errors too.

use std::path::PathBuf;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_app-cli");

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lcc-cli-lrcal-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("assets")).unwrap();
    std::fs::write(d.join("assets/ATTRIBUTION.md"), "| Path | Asset |\n|---|---|\n").unwrap();
    d
}

#[test]
fn add_refuses_hostile_files_without_panicking() {
    let repo = scratch("add");
    let inputs: [&[u8]; 6] =
        [b"", b"{bad", b"[]", br#"{"version":1,"make":"X","model":"Y","extra":true}"#, br#"{"version":99}"#, &[0xff, 0xfe, 0x00, 0x7b]];
    for (k, bytes) in inputs.iter().enumerate() {
        let file = repo.join(format!("in{k}.json"));
        std::fs::write(&file, bytes).unwrap();
        let o = Command::new(BIN).args(["calibrate-lightroom", "add", file.to_str().unwrap(), "--repo", repo.to_str().unwrap()]).output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(!o.status.success(), "input {k} accepted");
        assert!(!err.contains("panicked"), "input {k} panicked: {err}");
    }
    assert!(!repo.join("crates/camdb/data/calibrations").exists(), "nothing written for refused files");
    let o = Command::new(BIN).args(["calibrate-lightroom", "fit", "--raws"]).output().unwrap();
    assert!(!o.status.success());
    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn add_accepts_model_with_slash() {
    let repo = scratch("slash");
    let fm = lcc_raw::Mat3([[0.70, 0.18, 0.08], [0.28, 0.80, -0.08], [0.02, -0.15, 0.95]]);
    let cm = lcc_raw::Mat3([[0.9, -0.3, -0.05], [-0.5, 1.4, 0.1], [-0.1, 0.3, 0.6]]);
    let cal = lcc_camdb::calibration::Calibration {
        version: lcc_camdb::calibration::VERSION,
        make: "Xiaomi".into(),
        model: "Xiaomi 17 Ultra/2512BPNDAG".into(),
        aliases: vec![],
        editor: Default::default(),
        method: lcc_camdb::calibration::METHOD.into(),
        tool_version: "26.43.7".into(),
        contributor: Some("A Contributor".into()),
        licence: lcc_camdb::calibration::LICENCE.into(),
        camera: lcc_camdb::calibration::CameraModel { illuminants: [17, 21], color_matrix: [cm, cm], forward_matrix: [fm, fm], hue_sat_map: None },
        looks: vec![lcc_camdb::calibration::Look {
            name: "Camera Standard".into(),
            baseline_exposure_offset: 0.0,
            look_table: None,
            tone_curve: vec![],
            since_process: None,
        }],
        evidence: lcc_camdb::calibration::Evidence {
            photos: 10,
            samples: 1000,
            cct: vec![],
            hue_coverage: 0.5,
            white_points: 10,
            looks: vec![lcc_camdb::calibration::LookEvidence {
                name: "Camera Standard".into(),
                photos: 8,
                held_out: 2,
                before: Default::default(),
                after: Default::default(),
                since_process: None,
            }],
        },
    };
    let json = cal.to_json().unwrap();
    let file = repo.join("xiaomi.json");
    std::fs::write(&file, json).unwrap();
    let o = Command::new(BIN).args(["calibrate-lightroom", "add", file.to_str().unwrap(), "--repo", repo.to_str().unwrap()]).output().unwrap();
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    let dest = repo.join("crates/camdb/data/calibrations/XIAOMI_Xiaomi_17_Ultra_2512BPNDAG.json");
    assert!(dest.exists(), "calibration file was written");
    let attr = std::fs::read_to_string(repo.join("assets/ATTRIBUTION.md")).unwrap();
    assert!(attr.contains("XIAOMI_Xiaomi_17_Ultra_2512BPNDAG.json"));
    let _ = std::fs::remove_dir_all(&repo);
}
