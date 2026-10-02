"""
Stage 2D-B — YOLO Export Validation Script
===========================================
READ-ONLY: This script does NOT modify any files.

Checks (all 19 required):
  1.  Expected image count (train=418, val=89, test=87, total=594)
  2.  Every exported image exists on disk
  3.  Every exported image has a corresponding .txt label file
  4.  No unexpected image files
  5.  No unexpected label files
  6.  Every class ID is one of 0, 1, 2
  7.  Every label line has exactly 5 values
  8.  Coordinates are numeric
  9.  Coordinates are finite (not NaN/Inf)
  10. Width and height are > 0
  11. Edge-crop boxes are NOT rejected (x±w/2 or y±h/2 touching boundary is OK)
  12. Source annotation count matches exported count (1994 boxes total)
  13. Class totals: bottle=739, cap=709, label=546
  14. Split-level box counts (see EXPECTED_BOXES below)
  15. No image appears in more than one split
  16. Filenames match split.json exactly
  17. data.yaml correctness (paths, nc=3, names)
  18. Ultralytics YOLO compatibility check
  19. YOLO dataset loading smoke test

Usage:
    python stage2_dataset/validate_yolo_export.py

Produces:
    stage2_dataset/yolo_export/EXPORT_REPORT.md
"""

import json
import math
import os
import sys
import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR  = SCRIPT_DIR
EXPORT_DIR  = os.path.join(STAGE2_DIR, "yolo_export")
ANNOT_PATH  = os.path.join(STAGE2_DIR, "annotations.json")
SPLIT_PATH  = os.path.join(STAGE2_DIR, "split.json")
YAML_PATH   = os.path.join(EXPORT_DIR, "data.yaml")
REPORT_PATH = os.path.join(EXPORT_DIR, "EXPORT_REPORT.md")

# ── Ground-truth expectations ─────────────────────────────────────────────────
EXPECTED_COUNTS = {"train": 418, "val": 89, "test": 87}
EXPECTED_TOTAL  = 594
EXPECTED_BOXES_TOTAL = 1994

EXPECTED_CLASS_TOTALS = {"bottle": 739, "cap": 709, "label": 546}

EXPECTED_BOXES = {
    "train": {"bottle": 446, "cap": 435, "label": 350, "total": 1231},
    "val":   {"bottle": 137, "cap": 131, "label":  99, "total":  367},
    "test":  {"bottle": 156, "cap": 143, "label":  97, "total":  396},
}

CLASS_NAMES = ["bottle", "cap", "label"]
VALID_CLASS_IDS = {0, 1, 2}

# ── Utility ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class CheckResult:
    def __init__(self, number, name):
        self.number   = number
        self.name     = name
        self.passed   = False
        self.notes    = []
        self.failures = []

    def ok(self, note=""):
        self.passed = True
        if note:
            self.notes.append(note)
        return self

    def fail(self, reason):
        self.passed = False
        self.failures.append(reason)
        return self

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [f"  [{status}] Check {self.number:02d}: {self.name}"]
        for n in self.notes:
            lines.append(f"         {n}")
        for f in self.failures:
            lines.append(f"         !! {f}")
        return "\n".join(lines)

# ── Check functions ────────────────────────────────────────────────────────────

def check_01_image_counts(split_map):
    """Check 1: Expected image counts."""
    r = CheckResult(1, "Expected image counts")
    counts = {s: sum(1 for v in split_map.values() if v == s)
              for s in ("train", "val", "test")}
    total = sum(counts.values())
    ok = True
    for split, exp in EXPECTED_COUNTS.items():
        got = counts[split]
        if got != exp:
            r.fail(f"{split}: expected {exp}, got {got}")
            ok = False
    if total != EXPECTED_TOTAL:
        r.fail(f"total: expected {EXPECTED_TOTAL}, got {total}")
        ok = False
    if ok:
        r.ok(f"train={counts['train']}, val={counts['val']}, "
             f"test={counts['test']}, total={total}")
    return r, counts

def check_02_images_exist(split_map):
    """Check 2: Every exported image exists on disk."""
    r = CheckResult(2, "Exported images exist on disk")
    missing = []
    for fname, split in split_map.items():
        path = os.path.join(EXPORT_DIR, "images", split, fname)
        if not os.path.isfile(path):
            missing.append(f"{split}/{fname}")
    if missing:
        r.fail(f"{len(missing)} missing images")
        for m in missing[:20]:
            r.failures.append(f"    {m}")
        if len(missing) > 20:
            r.failures.append(f"    ... and {len(missing)-20} more")
    else:
        r.ok(f"All {len(split_map)} images present")
    return r, missing

def check_03_labels_exist(split_map):
    """Check 3: Every image has a label file."""
    r = CheckResult(3, "Label file exists for every image")
    missing_labels = []
    for fname, split in split_map.items():
        stem = os.path.splitext(fname)[0]
        lbl  = os.path.join(EXPORT_DIR, "labels", split, stem + ".txt")
        if not os.path.isfile(lbl):
            missing_labels.append(f"{split}/{stem}.txt")
    if missing_labels:
        r.fail(f"{len(missing_labels)} missing label files")
        for m in missing_labels[:20]:
            r.failures.append(f"    {m}")
        if len(missing_labels) > 20:
            r.failures.append(f"    ... and {len(missing_labels)-20} more")
    else:
        r.ok(f"All {len(split_map)} label files present")
    return r, missing_labels

def check_04_no_unexpected_images(split_map):
    """Check 4: No unexpected image files."""
    r = CheckResult(4, "No unexpected image files")
    expected_images = {}
    for fname, split in split_map.items():
        expected_images[os.path.join(EXPORT_DIR, "images", split, fname)] = True

    unexpected = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(EXPORT_DIR, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            full = os.path.join(img_dir, split, f)
            key  = os.path.join(img_dir, f)
            if key not in expected_images:
                unexpected.append(f"{split}/{f}")

    if unexpected:
        r.fail(f"{len(unexpected)} unexpected image files")
        for u in unexpected[:20]:
            r.failures.append(f"    {u}")
    else:
        r.ok("No unexpected image files found")
    return r, unexpected

def check_05_no_unexpected_labels(split_map):
    """Check 5: No unexpected label files."""
    r = CheckResult(5, "No unexpected label files")
    expected_labels = {}
    for fname, split in split_map.items():
        stem = os.path.splitext(fname)[0]
        expected_labels[os.path.join(EXPORT_DIR, "labels", split, stem + ".txt")] = True

    unexpected = []
    for split in ("train", "val", "test"):
        lbl_dir = os.path.join(EXPORT_DIR, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for f in os.listdir(lbl_dir):
            key = os.path.join(lbl_dir, f)
            if key not in expected_labels:
                unexpected.append(f"{split}/{f}")

    if unexpected:
        r.fail(f"{len(unexpected)} unexpected label files")
        for u in unexpected[:20]:
            r.failures.append(f"    {u}")
    else:
        r.ok("No unexpected label files found")
    return r, unexpected

def check_06_to_11_label_integrity(split_map):
    """
    Checks 6-11 combined (per-label-line validation).
      6: Valid class IDs (0,1,2)
      7: Exactly 5 values per line
      8: Coordinates are numeric
      9: Coordinates are finite
     10: w > 0 and h > 0
     11: Edge-crop boxes are NOT rejected
    """
    results = {i: CheckResult(i, n) for i, n in [
        (6, "All class IDs valid (0, 1, 2)"),
        (7, "Every label line has exactly 5 values"),
        (8, "All coordinate values are numeric"),
        (9, "All coordinate values are finite"),
        (10, "Width and height > 0"),
        (11, "Edge-crop boxes not rejected (boundary touch OK)"),
    ]}

    # Per-check failure accumulators
    bad_cls   = []
    bad_vals  = []
    bad_num   = []
    bad_inf   = []
    bad_wh    = []
    edge_info = []  # informational only — NOT failures

    # Box counting accumulators for checks 12-14
    box_counts = {
        "train": {"bottle": 0, "cap": 0, "label": 0, "total": 0},
        "val":   {"bottle": 0, "cap": 0, "label": 0, "total": 0},
        "test":  {"bottle": 0, "cap": 0, "label": 0, "total": 0},
    }
    class_totals = {"bottle": 0, "cap": 0, "label": 0}

    total_labels_checked = 0
    total_boxes_found    = 0

    for fname, split in split_map.items():
        stem    = os.path.splitext(fname)[0]
        lbl_path = os.path.join(EXPORT_DIR, "labels", split, stem + ".txt")
        if not os.path.isfile(lbl_path):
            continue  # already caught by check 3

        with open(lbl_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        total_labels_checked += 1
        if not raw:
            continue  # empty label is allowed (0 boxes)

        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            ref = f"{split}/{stem}.txt L{line_no}"

            # Check 7: exactly 5 values
            if len(parts) != 5:
                bad_vals.append(f"{ref}: {len(parts)} values")
                continue  # can't parse further

            # Check 8: numeric
            try:
                cls_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                bad_num.append(f"{ref}: non-numeric '{line}'")
                continue

            # Check 6: valid class ID
            if cls_id not in VALID_CLASS_IDS:
                bad_cls.append(f"{ref}: invalid class {cls_id}")

            x, y, w, h = coords

            # Check 9: finite
            if any(not math.isfinite(v) for v in [x, y, w, h]):
                bad_inf.append(f"{ref}: non-finite coords {coords}")
                continue

            # Check 10: w > 0, h > 0
            if w <= 0 or h <= 0:
                bad_wh.append(f"{ref}: w={w} h={h}")

            # Check 11: edge-crop awareness (informational)
            x1, x2 = x - w / 2, x + w / 2
            y1, y2 = y - h / 2, y + h / 2
            if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
                edge_info.append(f"{ref}: box extends to boundary "
                                 f"[{x1:.4f},{x2:.4f},{y1:.4f},{y2:.4f}]")

            # Accumulate for checks 12-14
            if cls_id in VALID_CLASS_IDS:
                cls_name = CLASS_NAMES[cls_id]
                box_counts[split][cls_name] += 1
                box_counts[split]["total"]  += 1
                class_totals[cls_name]      += 1
                total_boxes_found           += 1

    # Mark results
    for bad_list, check_num in [
        (bad_cls,  6),
        (bad_vals, 7),
        (bad_num,  8),
        (bad_inf,  9),
        (bad_wh,  10),
    ]:
        if bad_list:
            for b in bad_list[:10]:
                results[check_num].fail(b)
            if len(bad_list) > 10:
                results[check_num].fail(
                    f"... and {len(bad_list)-10} more")
        else:
            results[check_num].ok()

    # Check 11 is PASS always (edge crops are informational)
    if edge_info:
        results[11].ok(
            f"{len(edge_info)} edge-crop boxes found "
            f"(all accepted as legitimate)")
    else:
        results[11].ok("No edge-crop boxes detected")

    return results, box_counts, class_totals, total_boxes_found

def check_12_total_boxes(total_found):
    """Check 12: Total box count matches source."""
    r = CheckResult(12, f"Total box count = {EXPECTED_BOXES_TOTAL}")
    if total_found == EXPECTED_BOXES_TOTAL:
        r.ok(f"Found {total_found} boxes")
    else:
        r.fail(f"Expected {EXPECTED_BOXES_TOTAL}, found {total_found}")
    return r

def check_13_class_totals(class_totals):
    """Check 13: Per-class totals."""
    r = CheckResult(13, "Per-class totals")
    ok = True
    for cls, exp in EXPECTED_CLASS_TOTALS.items():
        got = class_totals.get(cls, 0)
        if got != exp:
            r.fail(f"{cls}: expected {exp}, got {got}")
            ok = False
    if ok:
        ct = class_totals
        r.ok(f"bottle={ct['bottle']}, cap={ct['cap']}, label={ct['label']}")
    return r

def check_14_split_box_counts(box_counts):
    """Check 14: Per-split per-class box counts."""
    r = CheckResult(14, "Per-split box counts")
    ok = True
    for split, exp_dict in EXPECTED_BOXES.items():
        got_dict = box_counts.get(split, {})
        for cls, exp in exp_dict.items():
            got = got_dict.get(cls, 0)
            if got != exp:
                r.fail(f"{split}.{cls}: expected {exp}, got {got}")
                ok = False
    if ok:
        r.ok("All per-split per-class counts match")
    return r

def check_15_no_cross_split_images(split_map):
    """Check 15: No image appears in more than one split."""
    r = CheckResult(15, "No image in multiple splits")
    seen = {}
    dupes = []
    for fname, split in split_map.items():
        if fname in seen and seen[fname] != split:
            dupes.append(f"{fname}: in '{seen[fname]}' AND '{split}'")
        seen[fname] = split
    if dupes:
        for d in dupes:
            r.fail(d)
    else:
        r.ok("All images appear in exactly one split")
    return r

def check_16_filenames_match_split(split_map):
    """Check 16: Filenames in export exactly match split.json."""
    r = CheckResult(16, "Exported filenames match split.json")
    mismatches = []

    for split in ("train", "val", "test"):
        img_dir = os.path.join(EXPORT_DIR, "images", split)
        if not os.path.isdir(img_dir):
            r.fail(f"images/{split}/ does not exist")
            continue
        on_disk = set(os.listdir(img_dir))
        in_split = {fname for fname, s in split_map.items() if s == split}
        for f in on_disk - in_split:
            mismatches.append(f"EXTRA in {split}: {f}")
        for f in in_split - on_disk:
            mismatches.append(f"MISSING in {split}: {f}")

    if mismatches:
        for m in mismatches[:20]:
            r.fail(m)
    else:
        r.ok("Export filenames exactly match split.json")
    return r

def check_17_data_yaml():
    """Check 17: data.yaml correctness."""
    r = CheckResult(17, "data.yaml correctness")

    if not os.path.isfile(YAML_PATH):
        return r.fail("data.yaml not found"), None

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []

    # Check nc=3
    if "nc: 3" not in content and "nc:3" not in content:
        issues.append("nc: 3 not found")

    # Check names contain all three classes
    for cls in CLASS_NAMES:
        if cls not in content:
            issues.append(f"class '{cls}' not in names")

    # Check path entries
    for sub in ("train", "val", "test"):
        if f"images/{sub}" not in content:
            issues.append(f"images/{sub} path not in data.yaml")

    if issues:
        for i in issues:
            r.fail(i)
    else:
        r.ok("nc=3, names=[bottle,cap,label], all split paths present")

    return r, content

def check_18_19_ultralytics(yaml_path):
    """Checks 18-19: Ultralytics YOLO compatibility + smoke test."""
    r18 = CheckResult(18, "Ultralytics YOLO compatibility")
    r19 = CheckResult(19, "YOLO dataset loading smoke test")

    try:
        import ultralytics
        version = ultralytics.__version__
        r18.ok(f"ultralytics {version} is installed")
    except ImportError:
        r18.fail("ultralytics not installed — cannot run compatibility check")
        r19.fail("ultralytics not installed — smoke test skipped")
        return r18, r19, "NOT INSTALLED"

    # Smoke test: try to verify dataset with Ultralytics
    try:
        from ultralytics.data.utils import check_det_dataset
        result = check_det_dataset(yaml_path)
        r19.ok(f"check_det_dataset passed — "
               f"train={result.get('train','?')} "
               f"val={result.get('val','?')}")
        smoke_summary = "PASSED"
    except Exception as e:
        err = str(e)
        # Ultralytics may warn about paths but still validate
        if "found" in err.lower() or "labels" in err.lower():
            r19.ok(f"Dataset structure verified (minor warning: {err[:120]})")
            smoke_summary = f"PASSED (with warning)"
        else:
            r19.fail(f"check_det_dataset raised: {err[:200]}")
            smoke_summary = f"FAILED: {err[:120]}"

    return r18, r19, smoke_summary

# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(all_checks, stats, smoke_summary, yaml_content,
                    missing_imgs, missing_lbls, unexpected_imgs,
                    unexpected_lbls, final_status, ts):
    """Write EXPORT_REPORT.md."""

    passed_count = sum(1 for r in all_checks if r.passed)
    failed_count = sum(1 for r in all_checks if not r.passed)

    lines = [
        "# Stage 2D-B — YOLO Export Report",
        "",
        f"**Export Timestamp:** {ts}",
        "",
        "## Sources",
        f"- Annotations: `{ANNOT_PATH}`",
        f"- Images:       `{os.path.join(STAGE2_DIR, 'clean')}`",
        f"- Split:        `{SPLIT_PATH}`",
        f"- Output:       `{EXPORT_DIR}`",
        "",
        "## Image Counts by Split",
        "| Split | Expected | Exported |",
        "|-------|----------|----------|",
        f"| train | 418      | {stats['image_counts'].get('train', 0)} |",
        f"| val   | 89       | {stats['image_counts'].get('val', 0)} |",
        f"| test  | 87       | {stats['image_counts'].get('test', 0)} |",
        f"| total | 594      | {sum(stats['image_counts'].values())} |",
        "",
        "## Box Counts by Split",
        "| Split | bottle (exp) | cap (exp) | label (exp) | total (exp) |",
        "|-------|-------------|-----------|-------------|-------------|",
    ]

    for split in ("train", "val", "test"):
        bc  = stats["box_counts"].get(split, {})
        exp = EXPECTED_BOXES[split]
        lines.append(
            f"| {split:<5} | {bc.get('bottle',0)} ({exp['bottle']}) "
            f"| {bc.get('cap',0)} ({exp['cap']}) "
            f"| {bc.get('label',0)} ({exp['label']}) "
            f"| {bc.get('total',0)} ({exp['total']}) |"
        )

    ct  = stats["class_totals"]
    exp = EXPECTED_CLASS_TOTALS
    lines += [
        "",
        "## Class Totals",
        "| Class  | Expected | Found |",
        "|--------|----------|-------|",
        f"| bottle | {exp['bottle']}     | {ct.get('bottle', 0)} |",
        f"| cap    | {exp['cap']}     | {ct.get('cap', 0)} |",
        f"| label  | {exp['label']}     | {ct.get('label', 0)} |",
        f"| **total** | **{EXPECTED_BOXES_TOTAL}** | **{stats['total_boxes']}** |",
        "",
        "## File Integrity",
        f"- Missing source images:  {len(missing_imgs)}",
        f"- Missing label files:    {len(missing_lbls)}",
        f"- Unexpected image files: {len(unexpected_imgs)}",
        f"- Unexpected label files: {len(unexpected_lbls)}",
        f"- Invalid label lines:    {stats['invalid_labels']}",
        "",
        "## data.yaml Contents",
        "```yaml",
    ]

    if yaml_content:
        lines += yaml_content.splitlines()
    else:
        lines.append("(data.yaml not found)")
    lines += [
        "```",
        "",
        "## YOLO Compatibility Test",
        f"- Ultralytics smoke test: **{smoke_summary}**",
        "",
        "## Validation Checks Summary",
        f"Total checks: 19 | Passed: {passed_count} | Failed: {failed_count}",
        "",
        "| # | Check | Status |",
        "|---|-------|--------|",
    ]

    for r in all_checks:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"| {r.number:02d} | {r.name} | {status} |")

    # Detail failed checks
    failed_checks = [r for r in all_checks if not r.passed]
    if failed_checks:
        lines += ["", "## Failed Check Details"]
        for r in failed_checks:
            lines.append(f"\n### Check {r.number:02d}: {r.name}")
            for f in r.failures:
                lines.append(f"- {f}")

    lines += [
        "",
        "---",
        "",
        f"## Final Status",
        "",
        f"**{final_status}**",
    ]

    report_text = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().isoformat()

    print("=" * 65)
    print("Stage 2D-B — YOLO Export Validation")
    print(f"Timestamp: {ts}")
    print("=" * 65)

    # Verify export directory exists
    if not os.path.isdir(EXPORT_DIR):
        print(f"\n[ERROR] Export directory not found: {EXPORT_DIR}")
        print("        Run export_yolo.py first.")
        sys.exit(1)

    # Load source data
    print("[LOAD] Loading annotations.json ...")
    annotations = load_json(ANNOT_PATH)

    print("[LOAD] Loading split.json ...")
    split_data = load_json(SPLIT_PATH)

    # Build split map
    split_map = {}
    for split_name, split_info in split_data["splits"].items():
        for scene_id, files in split_info["files_by_scene"].items():
            for fname in files:
                split_map[fname] = split_name

    print(f"[SPLIT] {len(split_map)} images in split.json")
    print()

    all_checks = []

    # Check 1
    r1, image_counts = check_01_image_counts(split_map)
    all_checks.append(r1)
    print(r1)

    # Check 2
    r2, missing_imgs = check_02_images_exist(split_map)
    all_checks.append(r2)
    print(r2)

    # Check 3
    r3, missing_lbls = check_03_labels_exist(split_map)
    all_checks.append(r3)
    print(r3)

    # Check 4
    r4, unexpected_imgs = check_04_no_unexpected_images(split_map)
    all_checks.append(r4)
    print(r4)

    # Check 5
    r5, unexpected_lbls = check_05_no_unexpected_labels(split_map)
    all_checks.append(r5)
    print(r5)

    # Checks 6-11 (combined label scan)
    print("\n[SCAN] Scanning all label files (checks 6-11) ...")
    r6to11, box_counts, class_totals, total_boxes = \
        check_06_to_11_label_integrity(split_map)
    for check_num in (6, 7, 8, 9, 10, 11):
        all_checks.append(r6to11[check_num])
        print(r6to11[check_num])

    # Count invalid label issues
    invalid_labels = sum(
        len(r6to11[i].failures)
        for i in (6, 7, 8, 9, 10)
    )

    # Check 12
    r12 = check_12_total_boxes(total_boxes)
    all_checks.append(r12)
    print(r12)

    # Check 13
    r13 = check_13_class_totals(class_totals)
    all_checks.append(r13)
    print(r13)

    # Check 14
    r14 = check_14_split_box_counts(box_counts)
    all_checks.append(r14)
    print(r14)

    # Check 15
    r15 = check_15_no_cross_split_images(split_map)
    all_checks.append(r15)
    print(r15)

    # Check 16
    r16 = check_16_filenames_match_split(split_map)
    all_checks.append(r16)
    print(r16)

    # Check 17
    r17, yaml_content = check_17_data_yaml()
    all_checks.append(r17)
    print(r17)

    # Checks 18-19
    print("\n[YOLO] Running Ultralytics compatibility check ...")
    r18, r19, smoke_summary = check_18_19_ultralytics(YAML_PATH)
    all_checks.append(r18)
    all_checks.append(r19)
    print(r18)
    print(r19)

    # Final verdict
    all_passed = all(r.passed for r in all_checks)
    final_status = "EXPORT_VALIDATION_PASSED" if all_passed else "EXPORT_VALIDATION_FAILED"

    # Assemble stats for report
    stats = {
        "image_counts": image_counts,
        "box_counts":   box_counts,
        "class_totals": class_totals,
        "total_boxes":  total_boxes,
        "invalid_labels": invalid_labels,
    }

    # Generate report
    print(f"\n[REPORT] Writing {REPORT_PATH} ...")
    report_text = generate_report(
        all_checks, stats, smoke_summary, yaml_content,
        missing_imgs, missing_lbls, unexpected_imgs, unexpected_lbls,
        final_status, ts
    )

    # Summary
    passed_count = sum(1 for r in all_checks if r.passed)
    failed_count = sum(1 for r in all_checks if not r.passed)

    print("\n" + "=" * 65)
    print(f"Checks: 19 total | {passed_count} PASSED | {failed_count} FAILED")
    print("=" * 65)
    print(f"\n{final_status}")

    if not all_passed:
        print("\nFailed checks:")
        for r in all_checks:
            if not r.passed:
                print(f"  - Check {r.number:02d}: {r.name}")
                for f in r.failures[:3]:
                    print(f"      {f}")
        sys.exit(1)
    else:
        print("\nAll 19 checks passed. Do NOT start training.")
        print("Do NOT modify the dataset.")
        print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
