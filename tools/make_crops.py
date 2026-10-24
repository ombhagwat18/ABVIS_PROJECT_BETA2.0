"""Bottle crops for the classifier, cut with the reviewed bottle mask (no ROI guessing, works for both cameras).

    python -m tools.make_crops --project enclosure_seg --target "Enclosure Crops"           # reviewed images only
    python -m tools.make_crops --project enclosure_seg --target "Enclosure Crops Smoke" --include-unreviewed
    python -m tools.make_crops --selftest

Every crop is the bottle's bounding box + a margin, widened to the classifier's aspect and resized to cache_wh, so the
new project's ROI is the whole crop (roi [0,0,W,H]) and `vision.train` needs no change. Labels come from the capture folder
(the person's own sort) and, where the reviewed image has NO label mask and the folder is not a missing-label folder, the
flag missing_label is added: a bottle without a label IS a missing label, whatever else is wrong with it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision import annotate as A                                        # noqa: E402
from vision import dataset as D                                         # noqa: E402

MARGIN = 0.10


def crop_box(points, shape, aspect: float, margin: float = MARGIN):
    """Normalized polygon -> pixel box [x0,y0,x1,y1] around the bottle, margin added, widened/heightened to `aspect` (w/h)."""
    h, w = shape[:2]
    xs, ys = np.array(points[0::2]) * w, np.array(points[1::2]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def cut(im, box, wh):
    """Crop with edge padding (the box may leave the frame), resize to wh = (W, H)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = im.shape[:2]
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return cv2.resize(im[y0 + pad:y1 + pad, x0 + pad:x1 + pad], wh, interpolation=cv2.INTER_AREA)


def build(slug: str, target: str, include_unreviewed: bool = False) -> dict:
    p = ROOT / "projects" / slug
    data = A.load_annotations(p / "annotations.json")
    meta = json.loads((p / "image_meta.json").read_text(encoding="utf-8"))
    iw, ih = D.CACHE_WH if hasattr(D, "CACHE_WH") else (216, 496)
    tmp = Path(tempfile.mkdtemp(prefix="crops_"))
    mapping, extra, n = {}, {}, 0
    for rel, e in data["images"].items():
        if not (e.get("reviewed") or include_unreviewed):
            continue
        bottle = next((q for q in e.get("polygons") or [] if q["cls"] == "bottle"), None)
        if bottle is None:
            continue
        folder = meta[rel]["folder"]
        im = cv2.imread(str(p / "images" / rel))
        out = cut(im, crop_box(bottle["points"], im.shape, iw / ih), (iw, ih))
        d = tmp / folder
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / Path(rel).name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if folder != "MissingLabel" and not any(q["cls"] == "label" for q in e.get("polygons") or []):
            extra[Path(rel).name] = "missing_label"
        n += 1
    names = {"GoodBottle": "GOOD", "DamageBottle": "damaged_bottle", "DamegeLable": "damaged_label",
             "MissingCap": "missing_cap", "SkewBottle": "skewed_bottle", "TiltedCap": "tilt_cap",
             "WaterLevel": "water_level"}
    mapping = {f.name: names[f.name] for f in tmp.iterdir() if f.is_dir()}
    name = D.create_project(target, "classification")
    D.use_project(name)
    got = D.run_import(tmp, mapping)
    cfg = D.load_config()
    cfg.update({"roi": [0, 0, iw, ih], "roi_frame": [iw, ih], "input_wh": [192, 448], "cache_wh": [iw, ih]})
    D.save_config(cfg)
    flagged = 0
    if extra:
        defects, labels = D.load_labels()
        by_name = {Path(r).name: r for r in labels}
        for fname, col in extra.items():
            rel = by_name.get(fname)
            if rel:
                D.apply_labels([rel], col, 1)
                flagged += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return {"project": name, "crops": n, "per_class": got, "missing_label_added": flagged}


def selftest():
    box = crop_box([0.4, 0.1, 0.5, 0.1, 0.5, 0.9, 0.4, 0.9], (1000, 2000), 216 / 496)
    w, h = box[2] - box[0], box[3] - box[1]
    assert abs(w / h - 216 / 496) < 1e-6 and box[1] < 100 and box[3] > 900
    im = np.zeros((100, 100, 3), np.uint8)
    assert cut(im, [-20, -10, 60, 120], (40, 80)).shape == (80, 40, 3)
    print("make_crops self-test ok")


# ---------------------------------------------------------------- crops of the CLEANED classification set (Experiment B)
CROP_WH = (288, 640)                          # bottle crop cache size; the classifier input is 256 x 576


def build_from_cls(target: str = "Enclosure Crops", src: str = "enclosure_cls",
                   ann: str = "projects/enclosure_seg/annotations_autoseed_backup.json", min_score: float = 0.8) -> dict:
    """One crop per image of the cleaned `src` project that has a trusted bottle outline in `ann` (a LOCALISER only: no label is
    taken from it). Labels (multi-label, incl. the retagged missing_label) are copied from src; ROI = the whole crop."""
    import csv
    ann_d = json.loads((ROOT / ann).read_text(encoding="utf-8"))
    poly_by_name = {}
    for rel, e in ann_d["images"].items():
        p = next((q for q in e.get("polygons") or [] if q["cls"] == "bottle" and q.get("score", 1) >= min_score), None)
        if p:
            poly_by_name[Path(rel).name] = p["points"]
    D.use_project(src)
    src_images, src_dir = D.IMAGE_ROOT, D.PROJECT_DIR
    with open(src_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    name = D.create_project(target, "classification")
    pdir = ROOT / "projects" / name
    kept, skipped = [], []
    for r in body:
        pts = poly_by_name.get(Path(r[0]).name)
        if pts is None:
            skipped.append(r[0])
            continue
        im = cv2.imread(str(src_images / r[0]))
        out = cut(im, crop_box(pts, im.shape, CROP_WH[0] / CROP_WH[1]), CROP_WH)
        dst = pdir / "images" / r[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(r)
    for d in head[1:-1]:
        (pdir / "images" / "-ve" / d).mkdir(parents=True, exist_ok=True)
    with open(pdir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(kept)
    cfg = {"roi": [0, 0, CROP_WH[0], CROP_WH[1]], "roi_frame": list(CROP_WH), "input_wh": [256, 576],
           "cache_wh": list(CROP_WH), "thresholds": {}}
    (pdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return {"project": name, "crops": len(kept), "skipped_no_bottle_outline": len(skipped),
            "good": sum(1 for r in kept if not any(v == "1" for v in r[1:-1]))}


# ---------------------------------------------------------------- crops of NEW captures from a segmentation model's bottle mask
def crop_with_model(weights: str, images, out_dir: Path, conf: float = 0.5, imgsz: int = 960) -> dict:
    """Every image -> one bottle crop (same crop_box / cut / CROP_WH as the classifier's training crops), the bottle found by
    a YOLO-seg model (class named "bottle"; the most confident one). An image with no bottle is listed, never guessed."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    cls = [k for k, v in m.names.items() if v == "bottle"]
    if not cls:
        raise SystemExit(f"{weights}: no 'bottle' class in {sorted(m.names.values())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done, missed = [], []
    for f in images:
        im = cv2.imread(str(f))
        r = m.predict(im, conf=conf, imgsz=imgsz, classes=cls, verbose=False)[0] if im is not None else None
        if r is None or r.masks is None or not len(r.boxes):
            missed.append(str(f))
            continue
        i = int(r.boxes.conf.argmax())
        pts = [float(v) for xy in r.masks.xyn[i] for v in xy]
        cv2.imwrite(str(out_dir / Path(f).name), cut(im, crop_box(pts, im.shape, CROP_WH[0] / CROP_WH[1]), CROP_WH),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        done.append(str(f))
    return {"weights": str(weights), "cropped": len(done), "no_bottle": missed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="enclosure_seg")
    ap.add_argument("--target", default="Enclosure Crops")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--from-cls", dest="from_cls", action="store_true", help="crop the cleaned enclosure_cls set (Experiment B)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", help="YOLO-seg weights: crop the images of --images into --out (new captures)")
    ap.add_argument("--images", help="folder of full frames (searched recursively for .jpg / .png)")
    ap.add_argument("--out", help="folder for the crops")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.model:
        files = sorted(f for f in Path(a.images).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(json.dumps(crop_with_model(a.model, files, Path(a.out)), indent=1))
    elif a.from_cls:
        print(build_from_cls(a.target))
    else:
        print(build(a.project, a.target, a.include_unreviewed))
