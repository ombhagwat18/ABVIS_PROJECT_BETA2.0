"""
Read-only heuristic audit of the new (unmerged) ~700-image bottle collection.
Does NOT touch Stage 1 dataset, labels.csv, or any project code.
Outputs: audit_data.csv, duplicate/near-duplicate groups, scene groups,
contact sheets, and a markdown summary report.

All "likely_*" flags are heuristic triage signals for MANUAL REVIEW ONLY,
not final labels.
"""
import os, sys, json, hashlib, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

SRC = Path(r"E:\MACHINE LEARNING PROJECT\Datasets\Bottle-train-OpenCV-main_vesion two\yolo_detection_dataset\images")
OUT = Path(r"E:\MACHINE LEARNING PROJECT\Datasets\Bottle-train-OpenCV-main_vesion two\FINAL_YEAR_BLACKBOOK\06_New_Dataset_Audit")
SHEETS = OUT / "contact_sheets"
EXAMPLES = OUT / "examples"
for p in (OUT, SHEETS, EXAMPLES):
    p.mkdir(parents=True, exist_ok=True)

# Known static UI-chrome exclusion zone for this capture rig (Iriun Webcam
# desktop screenshot, 1920x1080): title bar, hamburger menu, device dropdown,
# taskbar, and side letterbox bars. Content ROI is generous/conservative.
ROI = dict(top=70, bottom=95, left=95, right=95)

def content_roi(bgr):
    h, w = bgr.shape[:2]
    return bgr[ROI["top"]:h-ROI["bottom"], ROI["left"]:w-ROI["right"]], (ROI["left"], ROI["top"])

def dhash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def hamming(a, b):
    return bin(a ^ b).count("1")

def largest_contours(mask, min_area=1500, max_n=3):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts[:max_n]

def analyze(path):
    bgr_full = cv2.imread(str(path))
    h, w = bgr_full.shape[:2]
    roi, (ox, oy) = content_roi(bgr_full)
    rh, rw = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    result = {
        "width": w, "height": h,
        "n_object_blobs": 0,
        "bbox": None, "bbox_fill_frac": None, "aspect_wh": None,
        "angle_deg": None, "touches_top": False, "touches_bottom": False,
        "solidity": None,
    }

    # --- green mask (cap + label) -- the single most reliable anchor in
    # this dataset: cap and label are the only saturated-color regions
    # against a near-neutral wall/floor and a mostly-transparent bottle
    # body, so they localize cleanly even when edge-based body/base
    # segmentation is unreliable (smooth glass, faint wall smudges/seam). ---
    lower_g = np.array([40, 40, 40]); upper_g = np.array([90, 255, 255])
    gmask = cv2.inRange(hsv, lower_g, upper_g)
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    g_contours = largest_contours(gmask, min_area=300, max_n=8)

    # cluster green blobs by x-position first (candidate separate bottles),
    # then within the dominant cluster split top-most (cap) vs rest (label)
    def bx(c):
        x, _, bw, _ = cv2.boundingRect(c)
        return x, x + bw
    clusters = []
    for c in sorted(g_contours, key=lambda c: cv2.contourArea(c), reverse=True):
        cx0, cx1 = bx(c)
        placed = False
        for cl in clusters:
            mx0, mx1 = cl["xr"]
            pad = 60
            if cx0 - pad <= mx1 and cx1 + pad >= mx0:
                cl["xr"] = (min(mx0, cx0), max(mx1, cx1))
                cl["contours"].append(c)
                placed = True
                break
        if not placed:
            clusters.append({"xr": (cx0, cx1), "contours": [c]})
    result["n_object_blobs"] = max(len(clusters), 1)

    cap = None
    label = None
    if clusters:
        main_cluster = max(clusters, key=lambda cl: sum(cv2.contourArea(c) for c in cl["contours"]))
        cc = sorted(main_cluster["contours"], key=lambda c: cv2.boundingRect(c)[1])
        top_c = cc[0]
        x, y, bw, bh = cv2.boundingRect(top_c)
        cap = {"bbox": [x + ox, y + oy, bw, bh], "area": int(cv2.contourArea(top_c))}
        rect = cv2.minAreaRect(top_c)
        cap["angle_dev"] = round(min(abs(rect[-1]), 90 - abs(rect[-1])), 2)
        if len(cc) > 1:
            rest = sorted(cc[1:], key=lambda c: cv2.contourArea(c), reverse=True)
            lc = rest[0]
            x, y, bw, bh = cv2.boundingRect(lc)
            label = {"bbox": [x + ox, y + oy, bw, bh], "area": int(cv2.contourArea(lc))}
            rect = cv2.minAreaRect(lc)
            label["angle_dev"] = round(min(abs(rect[-1]), 90 - abs(rect[-1])), 2)
            hull = cv2.convexHull(lc)
            ha = cv2.contourArea(hull)
            label["solidity"] = round(cv2.contourArea(lc) / ha, 3) if ha > 0 else None

    result["cap"] = cap
    result["label"] = label

    # --- derived "bottle region" rectangle from cap+label anchors, used
    # for zoom/frame-fill and marker-spot search (a rough proxy, not a
    # true segmentation -- the neck/body/base below the label are not
    # directly detected) ---
    anchor_boxes = [b["bbox"] for b in (cap, label) if b]
    bottle_region = None
    if anchor_boxes:
        xs0 = min(b[0] for b in anchor_boxes) - 40
        xs1 = max(b[0] + b[2] for b in anchor_boxes) + 40
        ys0 = min(b[1] for b in anchor_boxes) - 20
        ys1 = h - ROI["bottom"]  # extend down to content-ROI bottom (base not directly detected)
        xs0, ys0 = max(xs0, 0), max(ys0, 0)
        xs1, ys1 = min(xs1, w), min(ys1, h)
        bottle_region = [int(xs0), int(ys0), int(xs1 - xs0), int(ys1 - ys0)]
        result["bbox"] = bottle_region
        result["bbox_fill_frac"] = round((bottle_region[2] * bottle_region[3]) / (rw * rh), 4)
        result["touches_top"] = ys0 <= (ROI["top"] + 3)
        cap_h = cap["bbox"][3] if cap else 0
        result["aspect_wh"] = round(cap_h / max(rh, 1), 4)  # reused field: "zoom" proxy (cap height / frame height)
        result["angle_deg"] = label["angle_dev"] if label else (cap["angle_dev"] if cap else None)
        result["solidity"] = label["solidity"] if label else None

    # --- dark marker-mark search: small near-black blobs, confined to the
    # derived bottle region (avoids the black window-chrome borders and
    # taskbar which sit outside the content ROI already) ---
    marker_like = 0
    if bottle_region:
        bx0, by0, bw_, bh_ = bottle_region
        sub_hsv = hsv[by0 - oy:by0 - oy + bh_, bx0 - ox:bx0 - ox + bw_]
        if sub_hsv.size:
            dark = cv2.inRange(sub_hsv, (0, 0, 0), (180, 255, 55))
            dcnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in dcnts:
                a = cv2.contourArea(c)
                if 15 <= a <= 900:
                    marker_like += 1
    result["marker_like_spots"] = marker_like

    # --- perceptual hash for near-duplicate detection ---
    small_gray = cv2.resize(gray, (256, 256))
    result["dhash"] = dhash(small_gray)

    # --- overall brightness (helps flag under/over-exposed / anomalous frames) ---
    result["mean_brightness"] = round(float(gray.mean()), 1)

    return result

def make_thumb(path, size=(220, 124)):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size[0]*3, size[1]*3))
    return im

def contact_sheet(files, out_path, cols=10, thumb=(160, 90), label_font=None):
    rows = math.ceil(len(files) / cols)
    W = cols * thumb[0]
    H = rows * (thumb[1] + 14)
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        try:
            im = Image.open(f).convert("RGB")
            im.thumbnail(thumb)
            x = (i % cols) * thumb[0]
            y = (i // cols) * (thumb[1] + 14)
            paste_x = x + (thumb[0] - im.width) // 2
            paste_y = y + (thumb[1] - im.height) // 2
            sheet.paste(im, (paste_x, paste_y))
            draw.text((x + 2, y + thumb[1] + 1), f.name if hasattr(f, "name") else str(f), fill=(200, 200, 200))
        except Exception as e:
            pass
    sheet.save(out_path)

def flag_records(records):
    finite = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))
    marker_vals = sorted(r["marker_like_spots"] for r in records if finite(r.get("marker_like_spots")))
    zoom_vals = sorted(r["aspect_wh"] for r in records if finite(r.get("aspect_wh")))
    solidity_vals = sorted(r["label"]["solidity"] for r in records
                            if r.get("label") and finite(r["label"].get("solidity")))
    def pct(vals, p):
        if not vals:
            return None
        k = int(len(vals) * p)
        return vals[min(k, len(vals) - 1)]
    marker_hi = pct(marker_vals, 0.97)
    zoom_hi = pct(zoom_vals, 0.90)
    # bottom 10% of label solidity, not a fixed absolute cutoff: this
    # label design prints a photo + QR code over the green background, so
    # solidity varies a lot with normal print content, not just damage --
    # an adaptive low-tail threshold is far less prone to false positives
    # than a fixed number (0.75 fixed flagged ~31% of the set, implausibly
    # high; see AUDIT_REPORT.md caveat).
    solidity_lo = pct(solidity_vals, 0.10)

    for r in records:
        flags = []
        cap, label = r.get("cap"), r.get("label")
        if cap is None and label is None:
            flags.append("ambiguous_no_anchor_detected")
        if cap is None and label is not None:
            flags.append("likely_missing_cap")
        if cap is not None and cap.get("angle_dev", 0) is not None and cap["angle_dev"] > 4:
            flags.append("likely_tilted_cap")
        if label is not None and label.get("angle_dev", 0) is not None and label["angle_dev"] > 3:
            flags.append("likely_skewed_label")
        if (label is not None and label.get("solidity") is not None
                and solidity_lo is not None and label["solidity"] <= solidity_lo):
            flags.append("likely_damaged_label")
        if r.get("n_object_blobs", 1) and r["n_object_blobs"] > 1:
            flags.append("likely_multiple_bottles")
        if r.get("aspect_wh") is not None and zoom_hi is not None and r["aspect_wh"] >= zoom_hi:
            flags.append("likely_close_up_view")
        if r.get("marker_like_spots") is not None and marker_hi is not None and r["marker_like_spots"] >= marker_hi:
            flags.append("candidate_black_marker_marking")
        if r.get("touches_top"):
            flags.append("frame_cropped_top")
        if not flags:
            flags.append("likely_good_normal_candidate")
        r["flags"] = flags
    return {"marker_hi_threshold": marker_hi, "zoom_hi_threshold": zoom_hi, "solidity_lo_threshold": solidity_lo}

def main():
    files = sorted(SRC.glob("*.png"), key=lambda p: p.name)
    print(f"Found {len(files)} images")

    records = []
    md5_groups = defaultdict(list)

    for i, f in enumerate(files):
        raw = f.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        md5_groups[md5].append(f.name)
        try:
            r = analyze(f)
        except Exception as e:
            r = {"error": str(e)}
        r["file"] = f.name
        r["bytes"] = len(raw)
        st = f.stat()
        r["mtime"] = st.st_mtime
        r["md5"] = md5
        records.append(r)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(files)}")

    # near-duplicate clustering via dHash hamming distance
    ordered = [r for r in records if "dhash" in r]
    ordered.sort(key=lambda r: r["mtime"])
    near_groups = []
    used = set()
    HAMMING_THRESH = 6
    for i, r in enumerate(ordered):
        if r["file"] in used:
            continue
        group = [r["file"]]
        used.add(r["file"])
        for j in range(i + 1, len(ordered)):
            r2 = ordered[j]
            if r2["file"] in used:
                continue
            if hamming(r["dhash"], r2["dhash"]) <= HAMMING_THRESH:
                group.append(r2["file"])
                used.add(r2["file"])
        if len(group) > 1:
            near_groups.append(group)

    # scene grouping by mtime gaps (new scene if gap > 8s) combined with
    # filename numeric sequence
    scenes = []
    cur = []
    prev_t = None
    for r in ordered:
        if prev_t is not None and (r["mtime"] - prev_t) > 8.0:
            if cur:
                scenes.append(cur)
            cur = []
        cur.append(r["file"])
        prev_t = r["mtime"]
    if cur:
        scenes.append(cur)

    thresholds = flag_records(records)

    # write per-image CSV
    fieldnames = ["file", "width", "height", "bytes", "mean_brightness",
                  "n_object_blobs", "bbox", "bbox_fill_frac", "aspect_wh",
                  "angle_deg", "touches_top", "touches_bottom", "solidity",
                  "cap", "label", "marker_like_spots", "flags", "md5", "dhash"]
    with open(OUT / "audit_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            for k in ("bbox", "cap", "label", "flags"):
                row[k] = json.dumps(row.get(k))
            w.writerow(row)

    with open(OUT / "audit_data.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1, default=str)

    with open(OUT / "duplicate_groups.json", "w", encoding="utf-8") as fh:
        json.dump({
            "exact_duplicates": {k: v for k, v in md5_groups.items() if len(v) > 1},
            "near_duplicate_groups": near_groups,
        }, fh, indent=2)

    with open(OUT / "scene_groups.json", "w", encoding="utf-8") as fh:
        json.dump([{"scene_id": i+1, "n_frames": len(g), "files": g} for i, g in enumerate(scenes)],
                   fh, indent=2)

    print("Exact dup groups:", sum(1 for v in md5_groups.values() if len(v) > 1))
    print("Near dup groups:", len(near_groups))
    print("Scenes:", len(scenes))

    # --- resolution / aspect-ratio distribution ---
    res_counts = defaultdict(int)
    ar_bucket = defaultdict(int)
    for r in records:
        res_counts[(r["width"], r["height"])] += 1
        ar = round(r["width"] / r["height"], 3)
        ar_bucket[ar] += 1

    # --- flag tally ---
    flag_counts = defaultdict(list)
    for r in records:
        for fl in r.get("flags", []):
            flag_counts[fl].append(r["file"])

    with open(OUT / "flag_tally.json", "w", encoding="utf-8") as fh:
        json.dump({k: {"count": len(v), "examples": v[:15]} for k, v in flag_counts.items()},
                   fh, indent=2)

    # --- contact sheets: full set in pages of 100 ---
    print("Building contact sheets...")
    for i in range(0, len(files), 100):
        page = files[i:i+100]
        contact_sheet(page, SHEETS / f"sheet_{i//100+1:02d}_{page[0].stem}-{page[-1].stem}.png")

    # --- one representative contact sheet per flag category ---
    for fl, names in flag_counts.items():
        sample_files = [SRC / n for n in names[:40]]
        if sample_files:
            contact_sheet(sample_files, EXAMPLES / f"{fl}.png")

    # --- markdown report ---
    lines = []
    lines.append("# New Dataset Audit -- Read-Only Report")
    lines.append("")
    lines.append(f"Source: `{SRC}`  ")
    lines.append(f"Generated by: `audit.py` (heuristic, OpenCV-based, no model inference)  ")
    lines.append("")
    lines.append("**This is a triage audit, not a labeling pass.** Every `likely_*` / "
                  "`candidate_*` flag below is a heuristic signal for a human to check -- "
                  "none of it was written to `labels.csv`, no images were moved/modified, "
                  "and the existing Stage 1 dataset (1,143 images) was not touched.")
    lines.append("")
    lines.append("## 1. Total image count")
    lines.append(f"- **{len(files)} images** found in `yolo_detection_dataset/images` (all `.png`), "
                 f"originally captured to `E:\\MACHINE LEARNING PROJECT\\BETA 2.O\\data\\raw\\images` "
                 f"and already copied into this repo (byte-identical, confirmed by MD5).")
    nums = sorted(int(f.stem.split('_')[1]) for f in files if f.stem.split('_')[1].isdigit())
    if nums:
        lines.append(f"- Filenames run `img_{nums[0]:03d}` .. `img_{nums[-1]:03d}`, but only "
                      f"{len(nums)} of the {nums[-1]-nums[0]+1} numbers in that range exist -- "
                      f"{nums[-1]-nums[0]+1-len(nums)} numbers are missing (likely discarded "
                      f"retakes during capture, not evidence of file loss).")
    lines.append("")
    lines.append("## 2. Resolution & aspect ratio")
    for (rw_, rh_), c in sorted(res_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {rw_}x{rh_}: {c} images ({c/len(files)*100:.1f}%)")
    lines.append("")
    lines.append("**Important:** every image is a full **1920x1080 desktop screenshot** of the "
                  "Iriun Webcam app (visible title bar, device dropdown, hamburger menu, and the "
                  "Windows taskbar are baked into the frame). This differs from the Stage 1 "
                  "capture rig (dedicated camera feed, calibrated ROI, near-black background). "
                  "The bottle itself occupies roughly 15-20% of the frame area. Before this "
                  "collection can be merged or annotated, it needs either a fixed crop to drop "
                  "the UI chrome, or a fresh ROI calibration -- it cannot reuse Stage 1's "
                  "`config.json` ROI as-is.")
    lines.append("")
    lines.append("## 3. Duplicate / near-duplicate detection")
    exact = [v for v in md5_groups.values() if len(v) > 1]
    lines.append(f"- Exact byte-identical duplicates: {len(exact)} groups "
                  f"({sum(len(v) for v in exact)} files involved).")
    lines.append(f"- Near-duplicate groups (perceptual hash, Hamming <= {6}): {len(near_groups)} groups, "
                  f"covering {sum(len(g) for g in near_groups)} files -- these are consecutive "
                  f"burst frames of the same bottle placement, not distinct samples.")
    lines.append("")
    lines.append("## 4. Scene / capture-session grouping")
    lines.append(f"- Inferred **{len(scenes)} capture scenes** by clustering consecutive frames "
                  f"whose file-modified timestamps are within 8s of each other (same idea as "
                  f"the Stage 1 project's scene-based train/val split -- consecutive frames of "
                  f"one bottle placement must not be split across train/val).")
    sizes = sorted((len(s) for s in scenes), reverse=True)
    lines.append(f"- Scene sizes range {min(sizes)}-{max(sizes)} frames, median {sizes[len(sizes)//2]}.")
    lines.append("")
    lines.append("## 5. One bottle vs. multiple bottles")
    n_multi = len(flag_counts.get("likely_multiple_bottles", []))
    lines.append(f"- **{n_multi}** images flagged `likely_multiple_bottles` "
                  f"(2+ disjoint cap/label color clusters at different x-positions). "
                  f"Low-recall heuristic: only catches a 2nd bottle if its cap/label is "
                  f"visibly green and separated -- manual spot-check recommended either way.")
    lines.append("")
    lines.append("## 6. Full-bottle vs. close-up / component views")
    n_closeup = len(flag_counts.get("likely_close_up_view", []))
    lines.append(f"- **{n_closeup}** images flagged `likely_close_up_view` (cap height takes an "
                  f"unusually large share of the frame -- top ~10% by a cap-height/frame-height "
                  f"zoom proxy, threshold={thresholds['zoom_hi_threshold']}).")
    lines.append("")
    lines.append("## 7-11. Heuristic defect-candidate flags (NOT final labels)")
    for fl in ["likely_good_normal_candidate", "likely_missing_cap", "likely_tilted_cap",
               "likely_skewed_label", "likely_damaged_label", "candidate_black_marker_marking",
               "frame_cropped_top", "ambiguous_no_anchor_detected"]:
        c = len(flag_counts.get(fl, []))
        lines.append(f"- `{fl}`: **{c}** images ({c/len(files)*100:.1f}%)")
    lines.append("")
    lines.append("Notes on precision:")
    lines.append("- `likely_damaged_label` is solidity-based (label contour area / convex-hull "
                  f"area, bottom 10% of the distribution, threshold<={thresholds['solidity_lo_threshold']}). "
                  "It correctly catches a torn hangtag/sticker seen during manual spot-checking "
                  "(`img_300`, `img_301`, solidity ~0.62). **But this label's print includes a "
                  "photo + QR code over the green background, so solidity varies a lot with "
                  "which part of that normal design is visible/glare-affected, not just with "
                  "physical damage** -- a fixed 0.75 cutoff flagged an implausible ~31% of the "
                  "set, so this was switched to an adaptive bottom-10% cutoff instead. Treat as "
                  "a shortlist to eyeball, similar confidence to the marker flag below, not a "
                  "validated detector.")
    lines.append("- `candidate_black_marker_marking`: manual spot-check of the top-scoring "
                  "examples (e.g. `img_554`-`img_567`) shows this **is picking up real dark "
                  "ink/scribble marks drawn on the label**, not just noise -- a better hit rate "
                  "than expected. Baseline speckle from water-droplet shadows/plastic ripples "
                  f"still exists (9-14 per image), so only the top ~3% by count "
                  f"(threshold={thresholds['marker_hi_threshold']}) were flagged and a few of "
                  "those may still be shadow, not marker -- confirm each one visually.")
    lines.append("- `likely_missing_cap` fires whenever a label was found but no separate cap-colored "
                  "blob was found above it -- can also mean the cap is out of frame, occluded, or "
                  "a non-green cap color. Manual check required.")
    lines.append("")
    lines.append("**Unplanned finding from manual spot-checking:** several images in the "
                  "`likely_tilted_cap` set (e.g. `img_625`, `img_626`, `img_666`, `img_671`-`img_673`) "
                  "show **a hand holding the bottle**, not a bottle standing free on the table. "
                  "This is a distinct capture condition (occlusion, different pose) that none of "
                  "the flags above were designed to catch -- worth a dedicated manual pass to "
                  "count how many hand-in-frame images exist before deciding whether they're "
                  "usable at all for a QC-line classifier trained on unoccluded bottles.")
    lines.append("")
    lines.append("## 12. Rotated / angled / different-distance examples")
    lines.append("- Captured via `likely_tilted_cap` / `likely_skewed_label` (angle deviation from "
                  "vertical/horizontal) and `likely_close_up_view` (distance proxy). See flag "
                  "tallies above.")
    lines.append("")
    lines.append("## 13. Ambiguous images requiring manual review")
    n_amb = len(flag_counts.get("ambiguous_no_anchor_detected", []))
    lines.append(f"- **{n_amb}** images had no detectable cap or label color blob at all "
                  f"(segmentation failed outright) -- these need eyes-on review before anything "
                  f"else can be inferred about them.")
    lines.append("")
    lines.append("## Files in this audit folder")
    lines.append("- `audit_data.csv` / `audit_data.json` -- one row per image, all measurements + flags")
    lines.append("- `duplicate_groups.json` -- exact and near-duplicate file groups")
    lines.append("- `scene_groups.json` -- inferred capture-session groups")
    lines.append("- `flag_tally.json` -- per-flag counts and example filenames")
    lines.append("- `contact_sheets/` -- full dataset, 100 images per sheet, in filename order")
    lines.append("- `examples/` -- one contact sheet per heuristic flag (up to 40 examples each)")
    lines.append("")
    lines.append("## Proposed structure once this collection is reviewed and approved for annotation")
    lines.append("```")
    lines.append("projects/<new-project-slug>/          # separate project, per dataset.py's")
    lines.append("                                       # multi-project design -- do NOT merge")
    lines.append("                                       # into the Stage 1 project's labels.csv")
    lines.append("  images/")
    lines.append("    _inbox/                            # all ~606 images start here, reviewed=0")
    lines.append("  labels.csv                           # created only once manual review begins")
    lines.append("  config.json                          # needs its OWN ROI calibration --")
    lines.append("                                       # different rig, different background")
    lines.append("```")
    lines.append("")
    lines.append("## Explicitly NOT done (per instructions)")
    lines.append("- No merge into the existing 1,143-image Stage 1 dataset")
    lines.append("- No writes to the existing `labels.csv`")
    lines.append("- No model training (classifier or YOLO)")
    lines.append("- No annotation files created")
    lines.append("- No project code changed")
    lines.append("")

    with open(OUT / "AUDIT_REPORT.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("Report written to", OUT / "AUDIT_REPORT.md")
    return records, md5_groups, near_groups, scenes, flag_counts, thresholds

if __name__ == "__main__":
    main()
