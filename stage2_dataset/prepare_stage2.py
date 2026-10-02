"""
Stage 2 dataset preparation: clean-image derivation + manual-review manifest.

Scope (per approved instructions, 2026-09-22):
  - Read-only against yolo_detection_dataset/images/ (the 606 audited images)
    and FINAL_YEAR_BLACKBOOK/06_New_Dataset_Audit/ (the accepted audit outputs).
  - Writes ONLY under stage2_dataset/ at the repo root.
  - Does NOT touch projects/<stage1>/labels.csv, does NOT train anything,
    does NOT create YOLO labels or final annotations.

Part A: copies each source image byte-for-byte into stage2_dataset/source/,
        and writes a cropped derivative (Iriun camera-viewport only, UI
        chrome removed) into stage2_dataset/clean/.
Part B: carries the audit's 40 scene groups and 18 near-duplicate groups
        forward unchanged into stage2_dataset/scene_map.json, instead of
        re-deriving or (worse) randomizing them.
Part C/D: builds stage2_dataset/review_manifest.csv (+ .json) with one row
        per image: filenames, scene id, near-dup group id, the audit's
        heuristic flags carried over as-is, and empty review fields
        (review_status=unreviewed, decision blank, notes blank) for a human
        to fill in later. No heuristic flag is treated as ground truth.

Run directly: python stage2_dataset/prepare_stage2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES = REPO_ROOT / "yolo_detection_dataset" / "images"
AUDIT_DIR = REPO_ROOT / "FINAL_YEAR_BLACKBOOK" / "06_New_Dataset_Audit"

OUT_DIR = REPO_ROOT / "stage2_dataset"
SOURCE_DIR = OUT_DIR / "source"
CLEAN_DIR = OUT_DIR / "clean"
MANIFEST_CSV = OUT_DIR / "review_manifest.csv"
MANIFEST_JSON = OUT_DIR / "review_manifest.json"
SCENE_MAP_OUT = OUT_DIR / "scene_map.json"

# Iriun Webcam v2.9.1 camera-viewport crop, measured against the desktop
# screenshot chrome (title bar, device-selector badge, left/right letterbox
# bars, Windows taskbar) and confirmed identical (pixel-exact border
# transitions) across img_001 / img_300 / img_600. This is NOT the Stage 1
# config.json ROI -- different rig, different background, deliberately not
# reused (see instructions).
CROP_X, CROP_Y, CROP_W, CROP_H = 70, 28, 1780, 1000

# Manually spot-checked in the audit report (AUDIT_REPORT.md section 7-11
# notes) -- no automatic hand-in-frame detector exists yet, so this is a
# seed list, not a full pass. Left in the manifest as a heuristic flag only.
KNOWN_HAND_HELD = {
    "img_625.png", "img_626.png", "img_666.png",
    "img_671.png", "img_672.png", "img_673.png",
}

FLAG_COLUMNS = {
    "likely_good_flag": "likely_good_normal_candidate",
    "likely_tilted_cap_flag": "likely_tilted_cap",
    "multiple_bottle_flag": "likely_multiple_bottles",
    "close_up_flag": "likely_close_up_view",
    "black_marker_flag": "candidate_black_marker_marking",
}
# label-defect flag is a union of the audit's two label-related heuristics
LABEL_DEFECT_SOURCE_FLAGS = {"likely_skewed_label", "likely_damaged_label"}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_audit_rows() -> dict[str, dict]:
    with open(AUDIT_DIR / "audit_data.csv", newline="", encoding="utf-8") as f:
        rows = {row["file"]: row for row in csv.DictReader(f)}
    return rows


def load_scene_map() -> dict[str, int]:
    scenes = json.loads((AUDIT_DIR / "scene_groups.json").read_text(encoding="utf-8"))
    out = {}
    for scene in scenes:
        for fn in scene["files"]:
            out[fn] = scene["scene_id"]
    return out


def load_dup_group_map() -> dict[str, int]:
    dup = json.loads((AUDIT_DIR / "duplicate_groups.json").read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for idx, group in enumerate(dup["near_duplicate_groups"], start=1):
        for fn in group:
            out.setdefault(fn, idx)  # first group a file appears in
    return out


def part_a_clean_images(files: list[str]) -> list[str]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = []
    for fn in files:
        src = SRC_IMAGES / fn
        dst_source = SOURCE_DIR / fn
        if not dst_source.exists() or md5sum(src) != md5sum(dst_source):
            shutil.copy2(src, dst_source)

        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        if (w, h) != (1920, 1080):
            mismatches.append(fn)
            continue
        crop = img[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
        cv2.imwrite(str(CLEAN_DIR / fn), crop)
    return mismatches


def build_manifest(files: list[str], audit_rows: dict, scene_map: dict, dup_map: dict) -> list[dict]:
    rows = []
    for fn in files:
        a = audit_rows[fn]
        flags = set(json.loads(a["flags"]))
        row = {
            "source_filename": fn,
            "clean_filename": fn,
            "scene_id": scene_map.get(fn, ""),
            "near_dup_group_id": dup_map.get(fn, ""),
            "multiple_bottle_flag": FLAG_COLUMNS["multiple_bottle_flag"] in flags,
            "close_up_flag": FLAG_COLUMNS["close_up_flag"] in flags,
            "likely_good_flag": FLAG_COLUMNS["likely_good_flag"] in flags,
            "likely_tilted_cap_flag": FLAG_COLUMNS["likely_tilted_cap_flag"] in flags,
            "likely_label_defect_flag": bool(flags & LABEL_DEFECT_SOURCE_FLAGS),
            "black_marker_flag": FLAG_COLUMNS["black_marker_flag"] in flags,
            "hand_held_flag": fn in KNOWN_HAND_HELD,
            "review_status": "unreviewed",
            "decision": "",
            "notes": "",
        }
        rows.append(row)
    return rows


def write_manifest(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_scene_map_copy() -> tuple[int, int]:
    scenes = json.loads((AUDIT_DIR / "scene_groups.json").read_text(encoding="utf-8"))
    dup = json.loads((AUDIT_DIR / "duplicate_groups.json").read_text(encoding="utf-8"))
    out = {
        "note": "Carried forward unchanged from the accepted audit "
                "(FINAL_YEAR_BLACKBOOK/06_New_Dataset_Audit/). Scene groups "
                "must stay intact for any future train/val/test split -- "
                "never redistribute consecutive frames of one scene.",
        "scenes": scenes,
        "near_duplicate_groups": dup["near_duplicate_groups"],
    }
    SCENE_MAP_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return len(scenes), len(dup["near_duplicate_groups"])


def selfcheck(files: list[str], rows: list[dict]) -> None:
    assert len(files) == 606, f"expected 606 source files, found {len(files)}"

    source_files = sorted(p.name for p in SOURCE_DIR.glob("*.png"))
    clean_files = sorted(p.name for p in CLEAN_DIR.glob("*.png"))
    assert source_files == sorted(files), "source/ does not mirror the input file set"
    assert clean_files == sorted(files), "clean/ does not mirror the input file set"

    for fn in files:
        assert md5sum(SRC_IMAGES / fn) == md5sum(SOURCE_DIR / fn), f"{fn}: source copy is not byte-identical"

    sample = clean_files[0]
    img = cv2.imread(str(CLEAN_DIR / sample))
    h, w = img.shape[:2]
    assert (w, h) == (CROP_W, CROP_H), f"clean image {sample} has wrong shape {img.shape}"

    assert len(rows) == 606, f"manifest has {len(rows)} rows, expected 606"
    assert all(r["review_status"] == "unreviewed" for r in rows), "manifest rows must start unreviewed"
    assert all(r["decision"] == "" for r in rows), "manifest decision must start blank"
    scene_ids = {r["scene_id"] for r in rows}
    assert "" not in scene_ids, "every image must have a scene_id from the audit"

    print("SELFCHECK OK")
    print(f"  source/: {len(source_files)} files, byte-identical to originals")
    print(f"  clean/:  {len(clean_files)} files, cropped to {CROP_W}x{CROP_H}")
    print(f"  manifest: {len(rows)} rows, all review_status=unreviewed, all scene_id populated")


def main() -> None:
    audit_rows = load_audit_rows()
    files = sorted(audit_rows.keys())
    scene_map = load_scene_map()
    dup_map = load_dup_group_map()

    mismatches = part_a_clean_images(files)
    if mismatches:
        raise RuntimeError(f"unexpected non-1920x1080 source images: {mismatches}")

    n_scenes, n_dup_groups = write_scene_map_copy()

    rows = build_manifest(files, audit_rows, scene_map, dup_map)
    write_manifest(rows)

    selfcheck(files, rows)

    n_review_needed = sum(
        1 for r in rows
        if not r["likely_good_flag"]
        or r["multiple_bottle_flag"] or r["close_up_flag"]
        or r["likely_tilted_cap_flag"] or r["likely_label_defect_flag"]
        or r["black_marker_flag"] or r["hand_held_flag"]
    )
    print(f"\nscenes carried forward: {n_scenes}")
    print(f"near-duplicate groups carried forward: {n_dup_groups}")
    print(f"images with >=1 non-good heuristic flag (priority for manual review): {n_review_needed}")


if __name__ == "__main__":
    main()
