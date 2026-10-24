"""Fast first-pass labels for the enclosure dataset: Bisleri-teal colour finds the cap and label, the old Stage-2
detector (when it agrees) or a cap/label-derived box finds the bottle, MobileSAM turns each box into a tight mask.

Every result is a PROPOSAL: a polygon with reviewed False, "seeded": True, "auto": True and a "score" (1 = passed the
geometry checks, <1 = look at it first). annotate.exportable never exports an unreviewed image, so nothing here can
reach training until a person has accepted it in the Annotation Studio (Review queue: lowest score first).

    python -m vision.autoseed --project enclosure_seg --preview 24       # picture of 24 random seeds, writes nothing
    python -m vision.autoseed --project enclosure_seg --run              # seed every image still without an entry
    python -m vision.autoseed --selftest                                 # synthetic image, no weights needed

Folder knowledge is used where it is certain: an image from the missing-cap folder gets no cap (the green tamper ring on
the bare neck looks like a cap: that is exactly the mistake the old detector made).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAM_WEIGHTS = ROOT / "models" / "pretrained" / "mobile_sam.pt"
DET_WEIGHTS = ROOT / "models" / "stage2_yolo" / "stage2_best.pt"
SCALE = 0.5                                   # work at half of 4K: masks of a ~150 px wide bottle, 1920 px frame
TEAL_LO, TEAL_HI = (70, 90, 60), (100, 255, 255)
PARTS = ("bottle", "cap", "label")
DEFECT_CLASSES = ("damaged_label", "damaged_bottle", "water_level", "tilt_cap", "missing_cap", "missing_label")
CLASSES = list(PARTS) + list(DEFECT_CLASSES)
FOLDER_DEFECT = {"GoodBottle": None, "DamageBottle": "damaged_bottle", "DamegeLable": "damaged_label",
                 "MissingCap": "missing_cap", "SkewBottle": "skewed_bottle", "TiltedCap": "tilt_cap",
                 "WaterLevel": "water_level"}


def teal_mask(small: np.ndarray) -> np.ndarray:
    m = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), TEAL_LO, TEAL_HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def teal_parts(small: np.ndarray, has_cap: bool = True):
    """-> (cap_box, label_box) as [x, y, w, h] in `small` pixels, either may be None."""
    h, w = small.shape[:2]
    n, _, st, _ = cv2.connectedComponentsWithStats(teal_mask(small))
    comps = sorted((tuple(int(v) for v in st[i]) for i in range(1, n) if st[i][4] >= 60 * (SCALE / 0.25) ** 2),
                   key=lambda c: c[1])
    cap = None
    if comps and has_cap and comps[0][2] < w * 0.2 and comps[0][3] < w * 0.16:
        cap = comps[0]
    rest = [c for c in comps if c is not cap]
    label = max(rest, key=lambda c: c[4]) if rest else None
    return cap, label


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def heuristic_bottle_box(cap, label, shape):
    """Box from the cap and/or label. Only a guide for SAM, which snaps to the real outline."""
    h, w = shape[:2]
    if cap and label:
        x0 = min(cap[0], label[0]) - 0.10 * label[2]
        x1 = max(cap[0] + cap[2], label[0] + label[2]) + 0.10 * label[2]
        y0 = cap[1] - 4
        y1 = min(h - 1, cap[1] + cap[2] * 6.4)
        y1 = max(y1, label[1] + label[3] + 0.9 * label[3])
        return [max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)]
    if cap:
        cx, bw = cap[0] + cap[2] / 2, cap[2] * 2.6
        return [max(0, cx - bw / 2), max(0, cap[1] - 4), min(w - 1, cx + bw / 2), min(h - 1, cap[1] + cap[2] * 6.4)]
    return None


def clean_mask(m: np.ndarray, box, pad: float = 0.06):
    """Clip to the (padded) box, keep the biggest blob, fill holes. -> uint8 mask or None."""
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    clip = np.zeros_like(m)
    clip[max(0, int(y0 - py)):int(y1 + py) + 1, max(0, int(x0 - px)):int(x1 + px) + 1] = 1
    m = (m.astype(np.uint8) & clip)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return filled


def to_polygon(mask: np.ndarray, eps: float = 0.0025):
    """mask -> flat normalized [x1,y1,...] list (largest contour, simplified) or None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    h, w = mask.shape[:2]
    return [round(float(v), 5) for p in c for v in (min(max(p[0] / w, 0.0), 1.0), min(max(p[1] / h, 0.0), 1.0))]


def bottle_score(mask: np.ndarray, cap, label) -> float:
    """1.0 = looks like a bottle (upright aspect, the cap / label sit inside it), lower = look at it first."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return 0.1
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    s = 1.0
    if not 2.0 <= bh / bw <= 6.5:
        s -= 0.4
    if len(xs) / float(bw * bh) < 0.55:
        s -= 0.3
    for part in (cap, label):
        if part is not None:
            cx, cy = int(part[0] + part[2] / 2), int(part[1] + part[3] / 2)
            if not mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)]:
                s -= 0.25
    return round(max(s, 0.05), 2)


class Seeder:
    """Loads MobileSAM (and the optional detector) once; seed(image_bgr, defect_hint) -> list of polygon dicts."""

    def __init__(self, use_detector: bool = True):
        from ultralytics import SAM, YOLO                      # lazy: the self-test and imports stay light
        self.sam = SAM(str(SAM_WEIGHTS))
        self.det = YOLO(str(DET_WEIGHTS)) if use_detector and DET_WEIGHTS.exists() else None

    def _sam_mask(self, small, box):
        r = self.sam(small, bboxes=[[float(v) for v in box]], verbose=False)[0]
        return r.masks.data[0].cpu().numpy().astype(np.uint8) if r.masks is not None and len(r.masks.data) else None

    def _detector_box(self, im, small, cap, label):
        if self.det is None:
            return None
        d = self.det(im, imgsz=1280, conf=0.25, verbose=False)[0]
        pts = [p for p in ((cap[0] + cap[2] / 2, cap[1] + cap[3] / 2) if cap else None,
                           (label[0] + label[2] / 2, label[1] + label[3] / 2) if label else None) if p]
        best = None
        for b, c in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.cls.cpu().numpy()):
            if int(c) != 0:
                continue
            b = [float(v) * SCALE for v in b]
            if pts and not all(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in pts):
                continue                                     # a "bottle" that does not hold the cap/label is wrong
            if best is None or (b[2] - b[0]) * (b[3] - b[1]) < (best[2] - best[0]) * (best[3] - best[1]):
                best = b                                     # tightest box that still holds the parts
        return best

    def seed(self, im: np.ndarray, defect: str | None = None) -> list:
        small = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        cap, label = teal_parts(small, has_cap=(defect != "missing_cap"))
        box = self._detector_box(im, small, cap, label)
        hbox = heuristic_bottle_box(cap, label, small.shape)
        if box is None or (hbox is not None and _iou(box, hbox) < 0.35):
            box = hbox or box
        elif hbox is not None:
            box = [box[0], box[1], box[2], max(box[3], hbox[3])]     # the detector box often stops above the base
        out = []
        if box is None:
            return out
        bm = self._sam_mask(small, box)
        bm = clean_mask(bm, box) if bm is not None else None
        if bm is None:
            return out
        score = bottle_score(bm, cap, label)
        poly = to_polygon(bm)
        if poly:
            out.append({"cls": "bottle", "points": poly, "score": score})
        for name, part in (("cap", cap), ("label", label)):
            if part is None:
                continue
            ctr = (min(part[1] + part[3] // 2, h - 1), min(part[0] + part[2] // 2, w - 1))
            if not bm[ctr]:
                continue                                       # outside the bottle: a reflection, not the part
            pad = 3
            pb = [part[0] - pad, part[1] - pad, part[0] + part[2] + pad, part[1] + part[3] + pad]
            pm = self._sam_mask(small, pb)
            pm = clean_mask(pm, pb, pad=0.1) if pm is not None else None
            p = to_polygon(pm) if pm is not None else None
            if p:
                out.append({"cls": name, "points": p, "score": 1.0})
        for o in out:
            o.update({"seeded": True, "auto": True, "model": "mobilesam+teal"})
        return out


# ----------------------------------------------------------------------------- project plumbing

def meta_path(slug: str) -> Path:
    return ROOT / "projects" / slug / "image_meta.json"


def build_meta(slug: str, src: Path) -> dict:
    """{rel: {folder, defect, camera}} for every image of the project (rel like '_inbox/snap_001.jpg').
    camera is a GUESS from the left/right edge brightness (A = dark wall on the left, B = scene on the left);
    replace it with the real id if the capture sessions are known."""
    meta = {}
    for d in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.jpg")):
            im = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_8)
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            w = g.shape[1]
            left, right = g[:, :w // 6].mean(), g[:, -w // 6:].mean()
            cam = "camA" if left < right - 10 else "camB" if right < left - 10 else "unknown"
            meta[f"_inbox/{f.name}"] = {"folder": d.name, "defect": FOLDER_DEFECT.get(d.name),
                                        "camera": cam, "camera_is_guess": True, "source": f"{d.name}/{f.name}"}
    meta_path(slug).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def run_project(slug: str, limit: int | None = None, preview: int = 0) -> dict:
    sys.path.insert(0, str(ROOT))
    from vision import annotate as A
    pdir = ROOT / "projects" / slug
    ann_path = pdir / "annotations.json"
    meta = json.loads(meta_path(slug).read_text(encoding="utf-8"))
    data = A.load_annotations(ann_path) if ann_path.exists() else A.init_annotations("segmentation", CLASSES)
    for c in CLASSES:
        if c not in data["classes"]:
            data["classes"].append(c)
    todo = [r for r in meta if r not in data["images"]]
    if preview:
        import random
        random.seed(3)
        todo = random.sample(todo, min(preview, len(todo)))
    elif limit:
        todo = todo[:limit]
    seeder = Seeder()
    t0, done, low, empty = time.time(), 0, 0, 0
    tiles = []
    for rel in todo:
        im = cv2.imread(str(pdir / "images" / rel))
        polys = seeder.seed(im, meta[rel].get("defect"))
        if not polys:
            empty += 1
        low += any(p["score"] < 0.8 for p in polys)
        entry = {"reviewed": False, "boxes": [], "polygons": [{k: v for k, v in p.items()} for p in polys],
                 "defect": meta[rel]["defect"], "camera": meta[rel]["camera"]}
        if preview:
            tiles.append(_draw(im, polys, rel))
        else:
            data["images"][rel] = entry
            done += 1
            if done % 100 == 0:
                A.save_annotations(ann_path, data)
                print(f"  {done}/{len(todo)}  {done / (time.time() - t0):.1f} img/s", flush=True)
    if preview:
        cols = 4
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - cols + 1, cols)]
        out = pdir / "autoseed_preview.jpg"
        cv2.imwrite(str(out), np.vstack(rows))
        return {"preview": str(out), "low_score": low, "empty": empty, "n": len(tiles)}
    A.save_annotations(ann_path, data)
    return {"seeded": done, "low_score": low, "empty": empty, "seconds": round(time.time() - t0)}


def _draw(im, polys, rel):
    small = cv2.resize(im, (450, 253))
    col = {"bottle": (0, 0, 255), "cap": (0, 255, 255), "label": (255, 0, 255)}
    for p in polys:
        pts = (np.array(p["points"]).reshape(-1, 2) * (450, 253)).astype(np.int32)
        cv2.polylines(small, [pts], True, col[p["cls"]], 2)
    cv2.putText(small, f"{rel[-8:-4]} {','.join(p['cls'][0] + str(p['score']) for p in polys)}", (4, 16), 0, 0.5, (0, 255, 0), 1)
    return small


def selftest():
    img = np.zeros((1080, 1920, 3), np.uint8) + 40
    cv2.rectangle(img, (900, 200), (1020, 260), (180, 200, 0), -1)          # teal-ish cap (BGR) near top
    cv2.rectangle(img, (880, 520), (1040, 640), (180, 200, 0), -1)          # label
    cap, label = teal_parts(cv2.resize(img, None, fx=SCALE, fy=SCALE))
    assert cap is not None and label is not None and cap[1] < label[1]
    box = heuristic_bottle_box(cap, label, (540, 960))
    assert box and box[0] < cap[0] and box[3] > label[1] + label[3]
    m = np.zeros((540, 960), np.uint8)
    m[100:400, 440:520] = 1
    assert bottle_score(m, cap, label) <= 1.0
    poly = to_polygon(clean_mask(m, [440, 100, 520, 400]))
    assert poly and len(poly) >= 6 and all(0 <= v <= 1 for v in poly)
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    print("autoseed self-test ok (geometry only; the SAM path is exercised by --preview)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--meta", help="build image_meta.json from this metadata folder first")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.project:
        if a.meta:
            print(len(build_meta(a.project, Path(a.meta))), "images tagged")
        if a.run or a.preview:
            print(run_project(a.project, a.limit, a.preview))
    else:
        ap.print_help()
