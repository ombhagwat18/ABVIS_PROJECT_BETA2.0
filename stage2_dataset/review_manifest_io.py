"""
Data layer for the Stage 2 review manifest -- load/save/ordering only, no
UI. Kept separate from review_app.py so the Stage 2 Annotation Studio can
reuse this module later (per instructions) without importing Tk.

Reads/writes ONLY:
  stage2_dataset/review_manifest.csv
  stage2_dataset/review_manifest.json

Never touches stage2_dataset/source/, stage2_dataset/clean/,
stage2_dataset/scene_map.json, the audit folder, or any Stage 1 project --
those are read-only inputs (clean/) or untouched entirely.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
MANIFEST_CSV = OUT_DIR / "review_manifest.csv"
MANIFEST_JSON = OUT_DIR / "review_manifest.json"
CLEAN_DIR = OUT_DIR / "clean"

FIELDNAMES = [
    "source_filename", "clean_filename", "scene_id", "near_dup_group_id",
    "multiple_bottle_flag", "close_up_flag", "likely_good_flag",
    "likely_tilted_cap_flag", "likely_label_defect_flag", "black_marker_flag",
    "hand_held_flag", "review_status", "decision", "notes",
]

FLAG_FIELDS = [
    "multiple_bottle_flag", "close_up_flag", "likely_tilted_cap_flag",
    "likely_label_defect_flag", "black_marker_flag", "hand_held_flag",
]

VALID_DECISIONS = {"", "KEEP", "REJECT", "AMBIGUOUS"}


def _to_bool(v) -> bool:
    return str(v).strip().lower() == "true"


def load_rows(path: Path = MANIFEST_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FLAG_FIELDS + ["likely_good_flag"]:
            r[k] = _to_bool(r[k])
        r["scene_id"] = int(r["scene_id"])
        r["near_dup_group_id"] = int(r["near_dup_group_id"]) if r["near_dup_group_id"] != "" else None
    return rows


def save_rows(rows: list[dict], csv_path: Path = MANIFEST_CSV, json_path: Path = MANIFEST_JSON) -> None:
    csv_rows = []
    for r in rows:
        row = dict(r)
        for k in FLAG_FIELDS + ["likely_good_flag"]:
            row[k] = str(bool(row[k]))
        row["near_dup_group_id"] = "" if row["near_dup_group_id"] is None else row["near_dup_group_id"]
        csv_rows.append(row)

    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(csv_rows)
    tmp_csv.replace(csv_path)

    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_json.replace(json_path)


def has_flag(row: dict) -> bool:
    return any(row[k] for k in FLAG_FIELDS)


def active_flags(row: dict) -> list[str]:
    return [k for k in FLAG_FIELDS if row[k]] + (["likely_good_flag"] if row["likely_good_flag"] else [])


def build_review_order(rows: list[dict]) -> list[int]:
    """Index order into `rows`: flagged images first (216 of them), grouped
    by scene so consecutive frames of one capture are reviewed together,
    then every remaining (heuristically "clean") image, also grouped by
    scene. Flags are review hints only -- this function never sets
    review_status or decision."""
    def sort_key(i):
        r = rows[i]
        return (r["scene_id"], r["source_filename"])

    flagged = sorted((i for i, r in enumerate(rows) if has_flag(r)), key=sort_key)
    rest = sorted((i for i, r in enumerate(rows) if not has_flag(r)), key=sort_key)
    return flagged + rest


def set_decision(row: dict, decision: str, notes: str | None = None) -> None:
    if decision not in VALID_DECISIONS - {""}:
        raise ValueError(f"invalid decision: {decision!r}")
    row["decision"] = decision
    row["review_status"] = "reviewed"
    if notes is not None:
        row["notes"] = notes
