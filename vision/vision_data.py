"""Unified vision dataset preparation: audit, manifest, YOLO-classification export, validation.

Read-only against the source datasets (projects/<slug>/ and stage2_dataset/). Everything it writes goes
under vision_dataset/, which is derived and can be deleted and rebuilt.

    python vision_data.py              # self-check on synthetic data in a temp folder (touches nothing real)
    python vision_data.py build        # audit both datasets -> vision_dataset/manifests/{unified_manifest.csv,dataset_report.json}
    python vision_data.py export-cls   # derived YOLO-classification tree (hard links, no pixel change, no extra disk)
    python vision_data.py export-cls --mode crop    # same, but lossless PNG crops of the calibrated bottle ROI
    python vision_data.py validate     # re-check manifest, exports, splits and that no checkpoint changed

What it deliberately does NOT do: train anything, touch annotations.json / labels.csv / any model file,
re-export the detection dataset (stage2_dataset/export_yolo.py owns that), or invent segmentation labels.
A bounding box is never turned into a polygon here.

Three tasks, three label sources:
  classification  projects/<slug>/labels.csv          one row per image, multi-label columns (see dataset.py)
  detection       stage2_dataset/annotations.json     boxes: bottle / cap / label; split.json; yolo_export/
  segmentation    polygons in an annotations.json     none exist yet -> reported as NOT READY
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vision import annotate
from vision import dataset as D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vision_dataset"
MANIFESTS = OUT / "manifests"
MANIFEST_CSV = MANIFESTS / "unified_manifest.csv"
REPORT_JSON = MANIFESTS / "dataset_report.json"
HASH_CACHE = MANIFESTS / "hash_cache.json"
CLS_OUT = OUT / "classification"
STAGE2 = ROOT / "stage2_dataset"
CLS_PROJECT = "om_bottle"
GOOD = "good"
SCENE_THRESHOLD = 4.0            # same test as dataset.scene_map: mean |diff| of 32x32 grey thumbnails
SPLITS = ("train", "val", "test")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
COLUMNS = ["image_id", "source_dataset", "image_path", "width", "height", "sha256", "scene_id", "split",
           "cls_labels", "cls_class", "cls_reviewed", "det_reviewed", "det_boxes", "det_classes",
           "seg_polygons", "seg_status", "duplicate_of", "notes"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ image signatures (cached)
def signatures(paths: list[Path], cache_path: Path | None = HASH_CACHE) -> dict:
    """{repo-relative path: {sha256, wh, thumb(32x32 float32)}}. Cached on (size, mtime) because decoding
    1,700 full-resolution frames takes minutes."""
    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    out, dirty = {}, False
    for p in paths:
        rel = p.relative_to(ROOT).as_posix() if ROOT in p.parents else p.as_posix()
        st = p.stat()
        c = cache.get(rel)
        if not c or c["size"] != st.st_size or c["mtime"] != int(st.st_mtime):
            raw = p.read_bytes()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                c = {"size": st.st_size, "mtime": int(st.st_mtime), "sha256": hashlib.sha256(raw).hexdigest(),
                     "wh": None, "thumb": None}
            else:
                t = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                c = {"size": st.st_size, "mtime": int(st.st_mtime), "sha256": hashlib.sha256(raw).hexdigest(),
                     "wh": [int(img.shape[1]), int(img.shape[0])], "thumb": base64.b64encode(t.tobytes()).decode()}
            cache[rel] = c
            dirty = True
        out[rel] = {"sha256": c["sha256"], "wh": c["wh"],
                    "thumb": None if c["thumb"] is None else
                    np.frombuffer(base64.b64decode(c["thumb"]), np.uint8).astype(np.float32).reshape(32, 32)}
    if dirty and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return out


# ------------------------------------------------------------------ pure helpers (covered by demo())
def cls_class(row: dict, defects: list[str]) -> str | None:
    """The single YOLO-classification class for a labels.csv row: the one defect, or 'good'.
    None = not expressible as one class (unreviewed, or more than one defect)."""
    if int(row.get("reviewed", 0)) != 1:
        return None
    on = [d for d in defects if int(row[d]) == 1]
    return GOOD if not on else (on[0] if len(on) == 1 else None)


def unify(groups: dict, links: list) -> dict:
    """Merge scene ids that are linked (the same physical scene filed under two class folders)."""
    parent = {g: g for g in set(groups.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in links:
        ra, rb = find(groups[a]), find(groups[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {p: find(g) for p, g in groups.items()}


def fix_leak(split_of: dict, groups: dict) -> tuple[dict, int]:
    """A scene must live in exactly one split. If a (merged) scene straddles splits, every image of it moves
    to the split that already holds most of its images (tie: the more held-out one, test > val > train)."""
    rank = {"train": 0, "val": 1, "test": 2}
    count = collections.defaultdict(collections.Counter)
    for p, s in split_of.items():
        count[groups[p]][s] += 1
    best = {g: max(c, key=lambda s: (c[s], rank[s])) for g, c in count.items()}
    out = {p: best[groups[p]] for p in split_of}
    return out, sum(out[p] != split_of[p] for p in split_of)


def val_before_test(split_of: dict, groups: dict, class_of: dict) -> tuple[dict, list]:
    """A class with too few scenes for three leak-free splits must still be in VAL (training needs it for model
    selection, and Ultralytics needs every class folder in val). If a class has test images but no val images,
    its test scenes move, whole, to val. Returns the new split and the classes this was done for."""
    out, done = dict(split_of), []
    for c in sorted(set(class_of.values())):
        have = {s: [p for p in out if class_of[p] == c and out[p] == s] for s in SPLITS}
        if not have["val"] and have["test"]:
            move = {groups[p] for p in have["test"]}
            for p in out:
                if groups[p] in move:
                    out[p] = "val"
            done.append(c)
    return out, done


def scene_leaks(split_of: dict, groups: dict) -> list:
    seen = collections.defaultdict(set)
    for p, s in split_of.items():
        seen[groups[p]].add(s)
    return sorted(g for g, s in seen.items() if len(s) > 1)


def box_problems(boxes: list[dict], classes: list[str], eps: float = 1e-6) -> list[str]:
    """Problems with one image's YOLO boxes (normalised centre x, y, w, h)."""
    bad, seen = [], set()
    for i, b in enumerate(boxes):
        if b.get("cls") not in classes:
            bad.append(f"box {i}: unknown class {b.get('cls')!r}")
            continue
        x, y, w, h = (float(b[k]) for k in "xywh")
        if not (w > 0 and h > 0):
            bad.append(f"box {i}: non-positive size")
        over = max(-(x - w / 2), -(y - h / 2), x + w / 2 - 1, y + h / 2 - 1)
        if not (0 <= x <= 1 and 0 <= y <= 1 and w <= 1 and h <= 1):
            bad.append(f"box {i}: outside the image")
        elif over > eps:                                 # centre inside, an edge past the frame: a bottle cut by the border
            bad.append(f"box {i}: edge past the frame by {over:.4f}")
        key = (b["cls"], round(x, 6), round(y, 6), round(w, 6), round(h, 6))
        if key in seen:
            bad.append(f"box {i}: exact duplicate")
        seen.add(key)
    return bad


def cross_folder_links(paths: list[str], sig: dict, threshold: float = SCENE_THRESHOLD) -> list:
    """Pairs of images in DIFFERENT class folders that are the same frame-run by dataset.scene_map's own
    test. dataset.scene_map only groups inside one folder, so these would otherwise be free to straddle a split."""
    have = [p for p in paths if sig[p]["thumb"] is not None]
    T = np.stack([sig[p]["thumb"].ravel() for p in have])
    folder = [p.rsplit("/", 1)[0] for p in have]
    links = []
    for i in range(len(have)):
        d = np.abs(T[i + 1:] - T[i]).mean(1)
        for j in np.nonzero(d < threshold)[0]:
            if folder[i] != folder[i + 1 + j]:
                links.append((have[i], have[i + 1 + int(j)]))
    return links


# ------------------------------------------------------------------ classification
def audit_classification(project: str = CLS_PROJECT) -> tuple[list[dict], dict]:
    pdir = ROOT / "projects" / project
    img_root = pdir / "images"
    with open(pdir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    defects = [c for c in rows[0] if c not in ("path", "reviewed")] if rows else []
    rep = {"project": project, "labels_csv": f"projects/{project}/labels.csv", "rows": len(rows), "defects": defects}
    malformed = [r["path"] for r in rows
                 if any(r[d] not in ("0", "1") for d in defects) or r["reviewed"] not in ("0", "1")]
    on_disk = sorted(p.relative_to(img_root).as_posix() for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXT)
    listed = [r["path"] for r in rows]
    rep.update(malformed_rows=malformed, missing_image_files=sorted(set(listed) - set(on_disk)),
               image_files_without_row=sorted(set(on_disk) - set(listed)),
               duplicate_rows=[p for p, n in collections.Counter(listed).items() if n > 1],
               reviewed=sum(r["reviewed"] == "1" for r in rows),
               per_defect={d: sum(r[d] == "1" for r in rows) for d in defects},
               good=sum(cls_class(r, defects) == GOOD for r in rows),
               multi_label_rows=sum(sum(r[d] == "1" for d in defects) > 1 for r in rows))
    rep["defects_with_zero_images"] = [d for d, n in rep["per_defect"].items() if n == 0]

    present = [p for p in listed if p in set(on_disk)]
    sig = signatures([img_root / p for p in present])
    key = lambda p: (img_root / p).relative_to(ROOT).as_posix()
    by_hash = collections.defaultdict(list)
    for p in present:
        by_hash[sig[key(p)]["sha256"]].append(p)
    dups = sorted(sorted(v) for v in by_hash.values() if len(v) > 1)
    rep["exact_duplicate_groups"] = dups
    rep["unreadable_images"] = [p for p in present if sig[key(p)]["wh"] is None]
    rep["image_sizes"] = {f"{w}x{h}": n for (w, h), n in collections.Counter(
        tuple(sig[key(p)]["wh"]) for p in present if sig[key(p)]["wh"]).items()}

    # scenes: the grouping training already uses (scenes.json), provided it still matches the image list
    scenes_json = pdir / "scenes.json"
    want = hashlib.md5(("|".join(sorted(present)) + f"|{SCENE_THRESHOLD}").encode()).hexdigest()
    groups = None
    if scenes_json.exists():
        c = json.loads(scenes_json.read_text())
        if c.get("key") == want:
            groups, rep["scene_source"] = c["map"], f"projects/{project}/scenes.json (cache key matches the current image list)"
    if groups is None:                                   # same algorithm as dataset.scene_map, on our own thumbnails
        groups, by_folder = {}, collections.defaultdict(list)
        for p in present:
            by_folder[p.rsplit("/", 1)[0]].append(p)
        for folder, items in sorted(by_folder.items()):
            prev, sid = None, 0
            for p in sorted(items, key=D._natkey):
                t = sig[key(p)]["thumb"]
                if t is None or prev is None or float(np.abs(t - prev).mean()) >= SCENE_THRESHOLD:
                    sid += 1
                groups[p] = f"{folder}#{sid}"
                prev = t
        rep["scene_source"] = "recomputed here with dataset.scene_map's rule (scenes.json missing or stale)"
    sig_rel = {p: sig[key(p)] for p in present}
    links = cross_folder_links(present, sig_rel)
    merged = unify(groups, links)
    rep["scenes"] = len(set(groups.values()))
    rep["cross_folder_same_scene_pairs"] = len(links)
    rep["cross_folder_linked_folders"] = sorted({" <-> ".join(sorted((a.rsplit("/", 1)[0], b.rsplit("/", 1)[0]))) for a, b in links})
    rep["scenes_after_merging_cross_folder_links"] = len(set(merged.values()))

    cls_of = {r["path"]: cls_class(r, defects) for r in rows}
    usable = [p for p in present if cls_of[p] is not None]
    tr, va, te = D.split_train_val_test(usable, val_frac=0.2, test_frac=0.15, seed=0, groups=merged)
    split_of = {**{p: "train" for p in tr}, **{p: "val" for p in va}, **{p: "test" for p in te}}
    split_of, moved = fix_leak(split_of, merged)
    cls_of = {r["path"]: cls_class(r, defects) for r in rows}
    split_of, rep["classes_given_val_instead_of_test"] = val_before_test(split_of, merged, {p: cls_of[p] for p in usable})
    rep["split_method"] = ("dataset.split_train_val_test (scene-based, per-folder quota, seed 0, val 0.20 / test 0.15) on "
                           "scenes merged across folders; a straddling scene moves whole to the split holding most of it")
    rep["images_moved_to_remove_cross_folder_leak"] = moved
    rep["scene_leaks"] = scene_leaks(split_of, merged)
    classes = sorted({c for c in cls_of.values() if c})
    rep["yolo_cls_classes"] = classes
    rep["split_counts"] = {s: sum(v == s for v in split_of.values()) for s in SPLITS}
    rep["split_per_class"] = {c: {s: sum(1 for p in usable if cls_of[p] == c and split_of[p] == s) for s in SPLITS} for c in classes}
    rep["scenes_per_class"] = {c: len({merged[p] for p in usable if cls_of[p] == c}) for c in classes}
    rep["classes_missing_from_a_split"] = {c: [s for s in SPLITS if v[s] == 0] for c, v in rep["split_per_class"].items()
                                           if any(v[s] == 0 for s in SPLITS)}
    rep["not_exportable_as_single_class"] = [p for p in present if cls_of[p] is None]

    first = {v[0]: v[1:] for v in dups}
    dup_of = {d: k for k, rest in first.items() for d in rest}
    out = []
    for r in rows:
        p = r["path"]
        s = sig_rel.get(p)
        on = [d for d in defects if r[d] == "1"]
        out.append({"image_id": f"cls:{p}", "source_dataset": "classification", "image_path": f"projects/{project}/images/{p}",
                    "width": s["wh"][0] if s and s["wh"] else "", "height": s["wh"][1] if s and s["wh"] else "",
                    "sha256": s["sha256"] if s else "", "scene_id": f"cls:{merged[p]}" if p in merged else "",
                    "split": split_of.get(p, ""), "cls_labels": "|".join(on) if on else (GOOD if r["reviewed"] == "1" else ""),
                    "cls_class": cls_of[p] or "", "cls_reviewed": r["reviewed"], "det_reviewed": "", "det_boxes": "",
                    "det_classes": "", "seg_polygons": 0, "seg_status": "none",
                    "duplicate_of": f"cls:{dup_of[p]}" if p in dup_of else "",
                    "notes": "" if s else "image file missing"})
    return out, rep


# ------------------------------------------------------------------ detection (Stage 2)
def audit_detection() -> tuple[list[dict], dict]:
    ann = annotate.load_annotations(STAGE2 / "annotations.json")
    split = json.loads((STAGE2 / "split.json").read_text())
    with open(STAGE2 / "review_manifest.csv", newline="", encoding="utf-8") as f:
        review = {r["clean_filename"]: r for r in csv.DictReader(f)}
    classes, images = ann["classes"], ann["images"]
    rep = {"annotations": "stage2_dataset/annotations.json", "annotations_sha256": sha256(STAGE2 / "annotations.json"),
           "split_sha256": sha256(STAGE2 / "split.json"), "task": ann["task"], "classes": classes, "images": len(images)}
    try:
        annotate.validate_annotations(ann)
        rep["schema_valid"] = True
    except annotate.AnnotationError as e:
        rep["schema_valid"] = f"INVALID: {e}"
    split_of, scene_of = {}, {}
    for s, block in split["splits"].items():
        for sid, files in (block.get("files_by_scene", {}) if isinstance(block, dict) else {}).items():
            for f in files:
                split_of.setdefault(f, []).append(s)
                scene_of[f] = str(sid)
    if not split_of:                                     # split.json layout without per-scene blocks: use the csv
        with open(STAGE2 / "split_manifest.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                split_of.setdefault(r["clean_filename"], []).append(r["split"].lower())
                scene_of[r["clean_filename"]] = r["scene_id"]
        rep["split_source"] = "stage2_dataset/split_manifest.csv"
    else:
        rep["split_source"] = "stage2_dataset/split.json"
    rep["images_in_two_splits"] = sorted(f for f, s in split_of.items() if len(set(s)) > 1)
    rep["annotated_without_split"] = sorted(set(images) - set(split_of))
    rep["split_without_annotation"] = sorted(set(split_of) - set(images))
    one = {f: s[0] for f, s in split_of.items()}
    rep["scene_leaks"] = scene_leaks({f: one[f] for f in images if f in one}, {f: scene_of[f] for f in images if f in one})
    rep["scene_id_disagrees_with_review_manifest"] = sorted(
        f for f in images if f in scene_of and f in review and review[f]["scene_id"] != scene_of[f])
    rep["split_counts"] = {s: sum(1 for f in images if one.get(f) == s) for s in SPLITS}
    rep["scenes"] = {s: len({scene_of[f] for f in images if one.get(f) == s}) for s in SPLITS}

    probs, edge, per_class, per_split = {}, [], collections.Counter(), {s: collections.Counter() for s in SPLITS}
    for f, e in images.items():
        p = box_problems(e.get("boxes", []), classes)
        edge += [float(q.rsplit(" ", 1)[1]) for q in p if "edge past the frame" in q]
        p = [q for q in p if "edge past the frame" not in q]
        if p:
            probs[f] = p
        for b in e.get("boxes", []):
            per_class[b["cls"]] += 1
            if one.get(f) in per_split:
                per_split[one[f]][b["cls"]] += 1
    rep.update(boxes=sum(per_class.values()), boxes_per_class=dict(per_class),
               boxes_per_split={s: dict(c) for s, c in per_split.items()}, box_problems=probs,
               boxes_with_an_edge_past_the_frame={"count": len(edge), "max_overshoot": max(edge, default=0.0),
                                                  "note": "centre inside the image; Ultralytics clips these at load"},
               images_without_boxes=sorted(f for f, e in images.items() if not e.get("boxes")),
               images_not_reviewed=sorted(f for f, e in images.items() if not e.get("reviewed")),
               polygons=sum(len(e.get("polygons", [])) for e in images.values()),
               images_without_label_box=sum(1 for e in images.values() if not any(b["cls"] == "label" for b in e["boxes"])),
               images_with_several_bottles=sum(1 for e in images.values() if sum(b["cls"] == "bottle" for b in e["boxes"]) > 1))

    # the existing YOLO export must be exactly what annotations.json + split.json say
    exp, ex = STAGE2 / "yolo_export", {"present": (STAGE2 / "yolo_export").exists()}
    if ex["present"]:
        ids = {c: i for i, c in enumerate(classes)}
        wrong_lbl, wrong_img, missing = [], [], []
        for f, e in images.items():
            s = one.get(f)
            lp, ip = exp / "labels" / s / (Path(f).stem + ".txt"), exp / "images" / s / f
            if not lp.exists() or not ip.exists():
                missing.append(f)
                continue
            want = sorted((ids[b["cls"]], round(b["x"], 5), round(b["y"], 5), round(b["w"], 5), round(b["h"], 5)) for b in e["boxes"])
            got = []
            for ln in lp.read_text().split("\n"):
                if ln.strip():
                    q = ln.split()
                    got.append((int(q[0]), *(round(float(v), 5) for v in q[1:5])))
            if sorted(got) != want:
                wrong_lbl.append(f)
            if ip.stat().st_size != (STAGE2 / "clean" / f).stat().st_size or sha256(ip) != sha256(STAGE2 / "clean" / f):
                wrong_img.append(f)
        on_disk = {s: sorted(p.name for p in (exp / "images" / s).glob("*")) for s in SPLITS if (exp / "images" / s).exists()}
        extra = sorted(n for s, names in on_disk.items() for n in names if one.get(n) != s)
        yaml_txt = (STAGE2 / "data.yaml").read_text()
        ex.update(missing_files=missing, label_files_differ_from_annotations=wrong_lbl, images_differ_from_clean=wrong_img,
                  unexpected_or_misplaced_images=extra, images_on_disk={s: len(v) for s, v in on_disk.items()},
                  portable_data_yaml="stage2_dataset/data.yaml",
                  data_yaml_names_ok=("names: ['bottle', 'cap', 'label']" in yaml_txt and "nc: 3" in yaml_txt))
    rep["yolo_export"] = ex

    sig = signatures([STAGE2 / "clean" / f for f in sorted(review) if (STAGE2 / "clean" / f).exists()])
    rep["image_sizes"] = {f"{w}x{h}": n for (w, h), n in collections.Counter(
        tuple(v["wh"]) for v in sig.values() if v["wh"]).items()}
    by_hash = collections.defaultdict(list)
    for k, v in sig.items():
        by_hash[v["sha256"]].append(Path(k).name)
    rep["exact_duplicate_groups"] = sorted(sorted(v) for v in by_hash.values() if len(v) > 1)
    rows = []
    for f in sorted(review):
        r, e, s = review[f], images.get(f), sig.get(f"stage2_dataset/clean/{f}")
        cc = collections.Counter(b["cls"] for b in e["boxes"]) if e else {}
        flags = [k[:-5] for k in r if k.endswith("_flag") and r[k] == "True"]
        rows.append({"image_id": f"det:{f}", "source_dataset": "detection", "image_path": f"stage2_dataset/clean/{f}",
                     "width": s["wh"][0] if s and s["wh"] else "", "height": s["wh"][1] if s and s["wh"] else "",
                     "sha256": s["sha256"] if s else "", "scene_id": f"det:{r['scene_id']}", "split": one.get(f, ""),
                     "cls_labels": "", "cls_class": "", "cls_reviewed": "", "det_reviewed": int(bool(e and e.get("reviewed"))) if e else "",
                     "det_boxes": sum(cc.values()) if e else "", "det_classes": "|".join(f"{c}:{cc[c]}" for c in classes if cc.get(c)),
                     "seg_polygons": len(e.get("polygons", [])) if e else 0, "seg_status": "none", "duplicate_of": "",
                     "notes": "; ".join(([f"review decision {r['decision']}"] if r["decision"] != "KEEP" else [])
                                        + ([("heuristic flags (not labels): " + ",".join(flags))] if flags else []))})
    return rows, rep


# ------------------------------------------------------------------ segmentation status + checkpoints
def segmentation_status() -> dict:
    """Polygons that actually exist, anywhere an annotations.json can hold them."""
    found = []
    for path in [STAGE2 / "annotations.json", *sorted((ROOT / "projects").glob("*/annotations.json"))]:
        try:
            a = annotate.load_annotations(path)
        except Exception as e:                           # noqa: BLE001 - reported, not hidden
            found.append({"file": path.relative_to(ROOT).as_posix(), "error": f"{type(e).__name__}: {e}"})
            continue
        polys = [(k, p) for k, e in a["images"].items() for p in e.get("polygons", [])]
        found.append({"file": path.relative_to(ROOT).as_posix(), "task": a["task"], "classes": a["classes"],
                      "images": len(a["images"]), "images_with_polygons": len({k for k, _ in polys}),
                      "polygons": len(polys), "polygons_per_class": dict(collections.Counter(p["cls"] for _, p in polys))})
    total = sum(f.get("polygons", 0) for f in found)
    return {"sources": found, "polygons_total": total, "ready": False,
            "why_not_ready": "no polygon annotations exist" if total == 0 else
                             "polygons exist but no validated segmentation export has been built"}


def checkpoints() -> dict:
    out = {}
    for p in [ROOT / "models/stage2_yolo/stage2_best.pt", ROOT / "yolov8n.pt", ROOT / "yolov8s.pt",
              *sorted((ROOT / "projects").glob("*/models/*/model.pt"))]:
        if p.exists():
            out[p.relative_to(ROOT).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
    return out


def cross_dataset(cls_rows: list[dict], det_rows: list[dict]) -> dict:
    """Do the two datasets share images? (They were captured separately; this proves it rather than assuming.)"""
    cs = signatures([ROOT / r["image_path"] for r in cls_rows if r["sha256"]])
    ds = signatures([ROOT / r["image_path"] for r in det_rows if r["sha256"]])
    same_bytes = sorted(set(v["sha256"] for v in cs.values()) & set(v["sha256"] for v in ds.values()))
    C = np.stack([v["thumb"].ravel() for v in cs.values() if v["thumb"] is not None])
    best = [float(np.abs(C - v["thumb"].ravel()).mean(1).min()) for v in ds.values() if v["thumb"] is not None]
    return {"byte_identical_images": len(same_bytes),
            "nearest_cross_dataset_thumbnail_distance": {"min": round(min(best), 2), "median": round(float(np.median(best)), 2)},
            "same_scene_threshold": SCENE_THRESHOLD, "images_within_threshold": int(sum(b < SCENE_THRESHOLD for b in best)),
            "images_with_both_classification_and_detection_labels": int(sum(b < SCENE_THRESHOLD for b in best)) + len(same_bytes)}


def build() -> dict:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    cls_rows, cls_rep = audit_classification()
    det_rows, det_rep = audit_detection()
    report = {"schema": "vision-dataset-report/1", "classification": cls_rep, "detection": det_rep,
              "segmentation": segmentation_status(), "cross_dataset": cross_dataset(cls_rows, det_rows),
              "checkpoints": checkpoints(),
              "manifest": {"file": "vision_dataset/manifests/unified_manifest.csv", "rows": len(cls_rows) + len(det_rows),
                           "classification_rows": len(cls_rows), "detection_rows": len(det_rows)}}
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(cls_rows + det_rows)
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------ YOLO classification export (derived)
def load_manifest() -> list[dict]:
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_cls(mode: str = "link") -> dict:
    """vision_dataset/classification/<split>/<class>/<image>. 'link' = hard links to the original files (no
    pixel change, no extra disk; falls back to a copy across volumes). 'crop' = lossless PNG of the calibrated
    bottle ROI via dataset.crop, the same crop the Stage 1 classifier is trained on."""
    rows = [r for r in load_manifest() if r["source_dataset"] == "classification" and r["cls_class"] and r["split"]
            and not r["duplicate_of"]]
    if CLS_OUT.exists():
        shutil.rmtree(CLS_OUT)                           # derived output only; sources are never under vision_dataset/
    cfg = json.loads((ROOT / "projects" / CLS_PROJECT / "config.json").read_text())
    files, linked, copied = {}, 0, 0
    for r in rows:
        src = ROOT / r["image_path"]
        dst = CLS_OUT / r["split"] / r["cls_class"] / (src.stem + (".png" if mode == "crop" else src.suffix))
        if dst.exists():                                 # same basename in two folders: keep both, unambiguously
            dst = dst.with_name(f"{hashlib.md5(r['image_path'].encode()).hexdigest()[:8]}_{dst.name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "crop":
            img = D.imread(src)
            cv2.imwrite(str(dst), D.crop(img, cfg.get("roi"), cfg.get("roi_frame")))
        else:
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
        files[dst.relative_to(CLS_OUT).as_posix()] = r["image_id"]
    info = {"mode": mode, "images": len(files), "hard_links": linked, "copies": copied,
            "classes": sorted({r["cls_class"] for r in rows}),
            "excluded_exact_duplicates": sum(1 for r in load_manifest() if r["source_dataset"] == "classification" and r["duplicate_of"]),
            "roi": cfg.get("roi"), "roi_frame": cfg.get("roi_frame"), "files": files}
    (CLS_OUT / "export_info.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------ validation
def validate() -> list[tuple[str, bool, str]]:
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
    rep = json.loads(REPORT_JSON.read_text())
    rows = load_manifest()
    c, d = rep["classification"], rep["detection"]
    cls = [r for r in rows if r["source_dataset"] == "classification"]
    det = [r for r in rows if r["source_dataset"] == "detection"]
    ck("1 classification: every row has valid 0/1 labels and an image file",
       not c["malformed_rows"] and not c["missing_image_files"] and not c["duplicate_rows"] and not c["unreadable_images"],
       f"{c['rows']} rows, malformed {len(c['malformed_rows'])}, missing files {len(c['missing_image_files'])}")
    ck("1b classification: every image is expressible as one YOLO class", not c["not_exportable_as_single_class"],
       f"{len(c['not_exportable_as_single_class'])} not exportable; multi-label rows {c['multi_label_rows']}")
    ex = d["yolo_export"]
    ck("2 detection: every annotated image has its YOLO label file, identical to annotations.json",
       ex.get("present") and not ex["missing_files"] and not ex["label_files_differ_from_annotations"]
       and not ex["images_differ_from_clean"] and not ex["unexpected_or_misplaced_images"],
       f"export images {ex.get('images_on_disk')}")
    seg = rep["segmentation"]
    ck("3 segmentation: polygons exist and are valid where required", seg["polygons_total"] > 0,
       f"{seg['polygons_total']} polygons -> NOT READY ({seg['why_not_ready']})")
    ck("4 class ids/names consistent", d["classes"] == ["bottle", "cap", "label"] and ex.get("data_yaml_names_ok")
       and d["schema_valid"] is True and not d["box_problems"],
       f"detection classes {d['classes']}; invalid boxes in {len(d['box_problems'])} images; "
       f"{d['boxes_with_an_edge_past_the_frame']['count']} boxes run past the frame edge (warning, clipped by Ultralytics); "
       f"cls classes {c['yolo_cls_classes']}")
    gs = collections.defaultdict(set)
    for r in rows:
        if r["split"]:
            gs[r["scene_id"]].add(r["split"])
    leaks = [g for g, s in gs.items() if len(s) > 1]
    ck("5 no train/val/test scene leakage (both datasets, re-derived from the manifest)", not leaks and not d["images_in_two_splits"],
       f"{len(gs)} scenes, straddling {len(leaks)}")
    dup_c = sum(len(g) - 1 for g in c["exact_duplicate_groups"])
    ck("6 no accidental image duplication", not d["exact_duplicate_groups"] and rep["cross_dataset"]["byte_identical_images"] == 0
       and all(r["duplicate_of"] for g in c["exact_duplicate_groups"] for r in cls if r["image_id"] in {f"cls:{x}" for x in g[1:]}),
       f"classification exact duplicates: {dup_c} (flagged duplicate_of, excluded from the export); detection: 0; shared between datasets: 0")
    broken = [r["image_path"] for r in rows if not (ROOT / r["image_path"]).exists()]
    ck("7 no broken paths in the manifest", not broken, f"{len(rows)} rows, broken {len(broken)}")
    ck("8 no malformed annotation files", d["schema_valid"] is True and not c["malformed_rows"]
       and not d["annotated_without_split"] and not d["split_without_annotation"] and not d["images_not_reviewed"],
       f"annotations.json schema valid={d['schema_valid']}")
    ck("9 dataset YAML/config correct (detection)", ex.get("data_yaml_names_ok"), "stage2_dataset/data.yaml: nc 3, names bottle/cap/label")
    # 9b classification export, if built
    info_p = CLS_OUT / "export_info.json"
    if info_p.exists():
        info = json.loads(info_p.read_text())
        by_id = {r["image_id"]: r for r in cls}
        bad, seen = [], collections.Counter()
        for rel, iid in info["files"].items():
            split, klass = rel.split("/")[:2]
            r, fp = by_id[iid], CLS_OUT / rel
            seen[iid] += 1
            if not fp.exists() or r["split"] != split or r["cls_class"] != klass:
                bad.append(rel)
            elif info["mode"] == "link" and sha256(fp) != r["sha256"]:
                bad.append(rel)
        on_disk = sum(1 for p in CLS_OUT.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
        train_classes = {rel.split("/")[1] for rel in info["files"] if rel.startswith("train/")}
        ck("9b classification export matches the manifest (split, class, bytes)",
           not bad and on_disk == len(info["files"]) and max(seen.values()) == 1 and train_classes == set(info["classes"]),
           f"{on_disk} files, mode {info['mode']}, mismatches {len(bad)}, classes in train {len(train_classes)}/{len(info['classes'])}")
        val_classes = {rel.split("/")[1] for rel in info["files"] if rel.startswith("val/")}
        test_classes = {rel.split("/")[1] for rel in info["files"] if rel.startswith("test/")}
        ck("9c classification: every class present in train AND val", train_classes == val_classes == set(info["classes"]),
           f"val {len(val_classes)}/{len(info['classes'])}; test has {len(test_classes)}/{len(info['classes'])} "
           f"(missing from test: {sorted(set(info['classes']) - test_classes) or 'none'} -> a name-mapped test evaluation is needed)")
    else:
        ck("9b classification export built", False, "run: python vision_data.py export-cls")
    now = checkpoints()
    changed = [k for k, v in rep["checkpoints"].items() if now.get(k, {}).get("sha256") != v["sha256"]]
    prov = json.loads((ROOT / "models/stage2_yolo/MODEL_PROVENANCE.json").read_text())
    prov_sha = json.dumps(prov)
    s2 = now.get("models/stage2_yolo/stage2_best.pt", {}).get("sha256", "?")
    ck("10 YOLOv8n Stage 2 checkpoint untouched", "models/stage2_yolo/stage2_best.pt" not in changed and s2 in prov_sha,
       f"sha256 {s2[:16]}... {'matches' if s2 in prov_sha else 'NOT FOUND IN'} MODEL_PROVENANCE.json")
    cls_ck = [k for k in now if k.startswith("projects/")]
    ck("11 classification checkpoints untouched", not [k for k in changed if k.startswith("projects/")]
       and set(cls_ck) == {k for k in rep["checkpoints"] if k.startswith("projects/")},
       f"{len(cls_ck)} model.pt files, changed since build: {len([k for k in changed if k.startswith('projects/')])}")
    ck("source annotation/split files unchanged since build",
       sha256(STAGE2 / "annotations.json") == d["annotations_sha256"] and sha256(STAGE2 / "split.json") == d["split_sha256"]
       and d["annotations_sha256"] in prov_sha, "annotations.json sha256 also matches MODEL_PROVENANCE.json")
    return checks


# ------------------------------------------------------------------ self-check
def demo():
    d = ["dent", "skew"]
    assert cls_class({"dent": "0", "skew": "0", "reviewed": "1"}, d) == GOOD
    assert cls_class({"dent": "1", "skew": "0", "reviewed": "1"}, d) == "dent"
    assert cls_class({"dent": "1", "skew": "1", "reviewed": "1"}, d) is None, "two defects are not one YOLO class"
    assert cls_class({"dent": "0", "skew": "0", "reviewed": "0"}, d) is None, "unreviewed must never export as good"
    g = {"a/1": "a#1", "a/2": "a#1", "b/1": "b#1", "b/2": "b#2", "c/1": "c#1"}
    m = unify(g, [("a/2", "b/1")])
    assert m["a/1"] == m["b/1"] and m["b/2"] != m["a/1"] and m["c/1"] == "c#1"
    sp = {"a/1": "train", "a/2": "train", "b/1": "test", "b/2": "val", "c/1": "train"}
    assert scene_leaks(sp, m) == [m["a/1"]]
    fixed, moved = fix_leak(sp, m)
    assert moved == 1 and fixed["a/1"] == fixed["a/2"] == fixed["b/1"] == "train" and not scene_leaks(fixed, m)
    vb, done = val_before_test({"r/1": "train", "r/2": "test", "q/1": "test", "z/1": "val"},
                               {"r/1": "r#1", "r/2": "r#2", "q/1": "r#2", "z/1": "z#1"},
                               {"r/1": "rare", "r/2": "rare", "q/1": "other", "z/1": "other"})
    assert done == ["rare"] and vb["r/2"] == vb["q/1"] == "val" and vb["r/1"] == "train", (vb, done)
    tie, _ = fix_leak({"a/1": "train", "b/1": "test"}, {"a/1": "g", "b/1": "g"})
    assert tie["a/1"] == tie["b/1"] == "test", "a tie goes to the more held-out split"
    ok = [{"cls": "x", "x": .5, "y": .5, "w": .2, "h": .2}]
    assert box_problems(ok, ["x"]) == []
    assert any("edge past the frame" in p for p in box_problems([{"cls": "x", "x": .95, "y": .5, "w": .2, "h": .2}], ["x"]))
    assert any("outside" in p for p in box_problems([{"cls": "x", "x": 1.2, "y": .5, "w": .2, "h": .2}], ["x"]))
    assert any("duplicate" in p for p in box_problems(ok + ok, ["x"]))
    assert any("unknown class" in p for p in box_problems([{"cls": "y", "x": .5, "y": .5, "w": .1, "h": .1}], ["x"]))
    assert any("non-positive" in p for p in box_problems([{"cls": "x", "x": .5, "y": .5, "w": 0, "h": .1}], ["x"]))
    # signatures + cross-folder links on synthetic frames in a temp folder
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.RandomState(0)
        base = rng.randint(0, 255, (64, 64), np.uint8)
        other = rng.randint(0, 255, (64, 64), np.uint8)
        for name, img in (("A/1.png", base), ("B/1.png", base.copy()), ("B/2.png", other)):
            (td / name).parent.mkdir(exist_ok=True)
            cv2.imwrite(str(td / name), img)
        s = signatures([td / "A/1.png", td / "B/1.png", td / "B/2.png"], cache_path=td / "cache.json")
        k = {Path(p).parent.name + "/" + Path(p).name: v for p, v in s.items()}
        assert k["A/1.png"]["sha256"] == k["B/1.png"]["sha256"] != k["B/2.png"]["sha256"] and k["A/1.png"]["wh"] == [64, 64]
        links = cross_folder_links(list(k), k)
        assert links == [("A/1.png", "B/1.png")], links
        assert signatures([td / "A/1.png"], cache_path=td / "cache.json")[next(iter(s))]["sha256"] == k["A/1.png"]["sha256"]
    print("ok  class mapping (unreviewed / multi-label never exported), cross-folder scene merge, leak fix, "
          "box checks, cached signatures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="demo", choices=["demo", "build", "export-cls", "validate"])
    ap.add_argument("--mode", default="link", choices=["link", "crop"])
    a = ap.parse_args(argv)
    if a.command == "demo":
        demo()
    elif a.command == "build":
        r = build()
        c, d = r["classification"], r["detection"]
        print(f"classification: {c['rows']} rows, {c['scenes']} scenes, split {c['split_counts']}")
        print(f"detection     : {d['images']} images, {d['boxes']} boxes {d['boxes_per_class']}, split {d['split_counts']}")
        print(f"segmentation  : {r['segmentation']['polygons_total']} polygons")
        print(f"manifest      : {r['manifest']['rows']} rows -> {MANIFEST_CSV.relative_to(ROOT)}")
    elif a.command == "export-cls":
        i = export_cls(a.mode)
        print(f"classification export: {i['images']} images, mode {i['mode']} (hard links {i['hard_links']}, copies {i['copies']}), "
              f"classes {i['classes']} -> {CLS_OUT.relative_to(ROOT)}")
    else:
        res = validate()
        for name, ok, detail in res:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        return 0 if all(ok for name, ok, _ in res if not name.startswith(("3 ", "9b classification export built"))) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
