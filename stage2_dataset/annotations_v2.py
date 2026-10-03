"""Stage 2 detection annotations, version 2: cap = the cap only.

Why (audit 2026-10-04): v1 (annotations.json) boxed `cap` as cap + neck finish down to the
tamper ring -- 703 of 709 cap boxes, in every split. That region still exists when the cap is
missing, so the v1 detector learned to call a bare neck "cap" and passed all 12 held-out
missing-cap bottles. v1 also has 6 class swaps (a cap boxed as `label` / `bottle`, a label boxed
as `cap`). No threshold fixes either.

v2 = v1 with
  1. the 6 class swaps below corrected (each checked by eye),
  2. every cap box tightened to the solid cap by colour: the cap is the first run of rows, from
     the top of the v1 box, that is >= half as green as the greenest row; the tamper ring is a
     separate, lower run and is left out. A box the rule cannot find a cap in keeps its v1 box
     and is listed as unchanged.
The split (split.json), images and the other boxes are untouched. annotations.json (v1) is
never written: its hash is pinned in models/stage2_yolo/MODEL_PROVENANCE.json.

AUTOMATIC CORRECTION, NOT HUMAN REVIEW. Every changed box is drawn in review_v2/ (old = orange,
new = magenta). The tightening was spot-checked by eye, not box by box; a human should page
through review_v2/ before any v2 model is deployed.

    python stage2_dataset/annotations_v2.py build     # annotations_v2.json + corrections_v2.json + review_v2/
    python stage2_dataset/annotations_v2.py export    # yolo_export_v2/ (same split, same images)
    python stage2_dataset/annotations_v2.py validate  # export == annotations_v2, split == v1, images == clean/
    python stage2_dataset/annotations_v2.py --selftest
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE / "annotations.json"
V2 = HERE / "annotations_v2.json"
LOG = HERE / "corrections_v2.json"
CLEAN = HERE / "clean"
REVIEW = HERE / "review_v2"
EXPORT_V1 = HERE / "yolo_export"
EXPORT_V2 = HERE / "yolo_export_v2"
SPLITS = ("train", "val", "test")

# (image, v1 class, predicate on the box, v2 class, why) -- each confirmed by looking at the image
CLASS_FIXES = [
    ("img_388.png", "label", "top", "cap", "cap boxed as label (box at bottle top)"),
    ("img_390.png", "label", "top", "cap", "cap boxed as label (box at bottle top)"),
    ("img_399.png", "bottle", "small", "cap", "cap boxed as bottle (small box at bottle top)"),
    ("img_557.png", "cap", "low", "label", "label boxed as cap (box at mid bottle)"),
    ("img_568.png", "cap", "low", "label", "label boxed as cap (box at mid bottle)"),
    ("img_724.png", "cap", "low", "label", "label boxed as cap (box at mid bottle)"),
]


def _px(b, W, H):
    return (max(0, int((b["x"] - b["w"] / 2) * W)), max(0, int((b["y"] - b["h"] / 2) * H)),
            min(W, int((b["x"] + b["w"] / 2) * W)), min(H, int((b["y"] + b["h"] / 2) * H)))


def tighten_cap(img, box):
    """v1 cap box -> (x1, y1, x2, y2) pixels around the solid cap only, or None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = _px(box, W, H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 65) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 40)).astype(np.uint8)
    rows = m.mean(1)
    on = np.where(rows > max(0.2, 0.5 * rows.max()))[0]
    if len(on) == 0:
        return None
    gap = max(1, int(0.01 * len(rows)))
    top = end = on[0]
    for r in on[1:]:
        if r - end > gap:
            break
        end = r
    cols = np.where(m[top:end + 1].mean(0) > 0.3)[0]
    if len(cols) == 0 or end - top < 3:
        return None
    pad = max(2, int(0.04 * (end - top + 1)))
    return (max(0, x1 + cols[0] - pad), max(0, y1 + top - pad),
            min(W, x1 + cols[-1] + pad + 1), min(H, y1 + end + pad + 1))


def _norm(x1, y1, x2, y2, W, H):
    return {"x": (x1 + x2) / 2 / W, "y": (y1 + y2) / 2 / H, "w": (x2 - x1) / W, "h": (y2 - y1) / H}


def _which(boxes, cls, pred):
    """Index of the v1 box a CLASS_FIXES row means (exactly one, or an error)."""
    big = [b for b in boxes if b["cls"] == "bottle" and b["h"] > 0.4]
    hits = []
    for i, b in enumerate(boxes):
        if b["cls"] != cls:
            continue
        for B in big:
            if b is B or abs(b["x"] - B["x"]) > B["w"] / 2:
                continue
            rel = (b["y"] - (B["y"] - B["h"] / 2)) / B["h"]
            if (pred == "top" and rel < 0.2) or (pred == "small" and rel < 0.2 and b["h"] < 0.3 * B["h"]) \
                    or (pred == "low" and rel > 0.35):
                hits.append(i)
                break
    if len(hits) != 1:
        raise ValueError(f"expected one {cls}/{pred} box, found {len(hits)}")
    return hits[0]


def build(v1=V1, v2=V2, log=LOG, clean=CLEAN, review=REVIEW, sheets=True) -> dict:
    data = json.loads(Path(v1).read_text(encoding="utf-8"))
    out = copy.deepcopy(data)
    out["version"] = 2
    out["derived_from"] = {"file": Path(v1).name, "sha256": hashlib.sha256(Path(v1).read_bytes()).hexdigest()}
    out["cap_definition"] = "cap only (tamper ring and neck excluded)"
    changes = []
    for img, cls, pred, new, why in CLASS_FIXES:
        if img not in out["images"]:
            continue
        i = _which(out["images"][img]["boxes"], cls, pred)
        out["images"][img]["boxes"][i]["cls"] = new
        changes.append({"image": img, "box": i, "kind": "class", "from": cls, "to": new, "why": why})
    unchanged = []
    for img, e in sorted(out["images"].items()):
        caps = [i for i, b in enumerate(e["boxes"]) if b["cls"] == "cap"]
        if not caps:
            continue
        im = cv2.imread(str(Path(clean) / img))
        if im is None:
            raise FileNotFoundError(Path(clean) / img)
        H, W = im.shape[:2]
        for i in caps:
            b = e["boxes"][i]
            t = tighten_cap(im, b)
            if t is None:
                unchanged.append({"image": img, "box": i})
                continue
            nb = _norm(*t, W, H)
            changes.append({"image": img, "box": i, "kind": "tighten",
                            "from": {k: round(b[k], 5) for k in "xywh"}, "to": {k: round(nb[k], 5) for k in "xywh"},
                            "height_ratio": round(nb["h"] / b["h"], 3)})
            b.update(nb)
    Path(v2).write_text(json.dumps(out, indent=2), encoding="utf-8")
    rep = {"v1_sha256": out["derived_from"]["sha256"],
           "v2_sha256": hashlib.sha256(Path(v2).read_bytes()).hexdigest(),
           "class_fixes": sum(c["kind"] == "class" for c in changes),
           "caps_tightened": sum(c["kind"] == "tighten" for c in changes),
           "caps_unchanged": unchanged, "changes": changes}
    Path(log).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if sheets:
        _sheets(data, out, changes, clean, review)
    return rep


def _sheets(v1, v2, changes, clean, review, per=12):
    """Contact sheets of every changed image: v1 boxes orange, v2 boxes magenta."""
    review = Path(review)
    review.mkdir(exist_ok=True)
    imgs = sorted({c["image"] for c in changes})
    tiles = []
    for k in imgs:
        im = cv2.imread(str(Path(clean) / k))
        H, W = im.shape[:2]
        for data, col, th in ((v1, (0, 165, 255), 6), (v2, (255, 0, 255), 3)):
            for b in data["images"][k]["boxes"]:
                x1, y1, x2, y2 = _px(b, W, H)
                cv2.rectangle(im, (x1, y1), (x2, y2), col, th)
                cv2.putText(im, b["cls"], (x1 + 4, y1 + 34), 0, 1.2, col, 3)
        t = cv2.resize(im, (534, 300))
        cv2.putText(t, k, (5, 28), 0, 0.9, (0, 0, 255), 2)
        tiles.append(t)
    for p in range(0, len(tiles), per):
        page = tiles[p:p + per]
        while len(page) % 3:
            page.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(review / f"page_{p // per + 1:03d}.jpg"),
                    np.vstack([np.hstack(page[i:i + 3]) for i in range(0, len(page), 3)]))


def export(v2=V2, out=EXPORT_V2):
    """The existing exporter, pointed at v2 (its paths are module globals read at call time)."""
    import export_yolo as E
    E.ANNOT_PATH, E.EXPORT_DIR = str(v2), str(out)
    E.main()


def validate(v2=V2, v1_export=EXPORT_V1, v2_export=EXPORT_V2, clean=CLEAN) -> list:
    """Errors (empty = ok): same files per split as v1, images byte-equal to clean/, labels == v2."""
    errs = []
    data = json.loads(Path(v2).read_text(encoding="utf-8"))
    ids = {"bottle": 0, "cap": 1, "label": 2}
    for s in SPLITS:
        a = sorted(p.name for p in (Path(v1_export) / "images" / s).glob("*.png"))
        b = sorted(p.name for p in (Path(v2_export) / "images" / s).glob("*.png"))
        if a != b:
            errs.append(f"{s}: v2 has {len(b)} images, v1 {len(a)}, or different files")
        for name in b:
            img = Path(v2_export) / "images" / s / name
            if img.stat().st_size != (Path(clean) / name).stat().st_size:
                errs.append(f"{s}/{name}: image differs from clean/")
            got = sorted(tuple(round(float(v), 4) for v in ln.split()) for ln in
                         (Path(v2_export) / "labels" / s / (Path(name).stem + ".txt")).read_text().splitlines() if ln.strip())
            want = sorted((float(ids[x["cls"]]), round(x["x"], 4), round(x["y"], 4), round(x["w"], 4), round(x["h"], 4))
                          for x in data["images"][name]["boxes"])
            if len(got) != len(want) or any(abs(g - w) > 2e-4 for G, Wt in zip(got, want) for g, w in zip(G, Wt)):
                errs.append(f"{s}/{name}: labels differ from annotations_v2.json")
    return errs


def selftest():
    import tempfile
    W, H = 400, 300
    img = np.full((H, W, 3), 230, np.uint8)
    cv2.rectangle(img, (150, 60), (250, 30), (150, 160, 30), -1)        # BGR teal-green cap, rows 30-60
    cv2.rectangle(img, (140, 70), (260, 76), (150, 160, 30), -1)        # tamper ring, rows 70-76
    v1box = {"cls": "cap", **_norm(130, 20, 270, 120, W, H)}            # loose v1 box: cap + ring + neck
    t = tighten_cap(img, v1box)
    assert t is not None and 26 <= t[1] <= 30 and 60 <= t[3] <= 66, t       # stops above the ring
    assert 145 <= t[0] <= 150 and 250 <= t[2] <= 256, t
    blank = np.full((H, W, 3), 230, np.uint8)
    assert tighten_cap(blank, v1box) is None, "no cap -> keep v1 box, never invent one"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "clean").mkdir()
        cv2.imwrite(str(d / "clean" / "img_388.png"), img)
        cv2.imwrite(str(d / "clean" / "x.png"), img)
        bottle = {"cls": "bottle", **_norm(120, 15, 280, 290, W, H)}
        v1 = {"schema_version": 1, "task": "detection", "classes": ["bottle", "cap", "label"], "images": {
            "img_388.png": {"reviewed": True, "polygons": [], "boxes": [dict(bottle),
                                                                         {"cls": "label", **_norm(130, 20, 270, 80, W, H)}]},
            "x.png": {"reviewed": True, "polygons": [], "boxes": [dict(bottle), dict(v1box)]}}}
        (d / "a.json").write_text(json.dumps(v1))
        before = (d / "a.json").read_bytes()
        rep = build(d / "a.json", d / "b.json", d / "c.json", d / "clean", d / "rev", sheets=True)
        assert (d / "a.json").read_bytes() == before, "v1 must never be written"
        assert rep["class_fixes"] == 1 and rep["caps_tightened"] == 2, rep
        v2 = json.loads((d / "b.json").read_text())
        assert [b["cls"] for b in v2["images"]["img_388.png"]["boxes"]] == ["bottle", "cap"]
        assert v2["images"]["x.png"]["boxes"][1]["h"] < 0.5 * v1box["h"]
        assert v2["images"]["x.png"]["boxes"][0] == bottle, "non-cap boxes untouched"
        assert list((d / "rev").glob("page_*.jpg"))
    print("ok  tighten stops above the tamper ring, no cap -> unchanged, class fix applied, v1 untouched")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if cmd == "--selftest":
        selftest()
    elif cmd == "build":
        r = build()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in r.items()})
    elif cmd == "export":
        sys.path.insert(0, str(HERE))
        export()
    elif cmd == "validate":
        e = validate()
        print("ok  v2 export: same split as v1, images == clean/, labels == annotations_v2.json" if not e
              else "\n".join(e[:30]))
        sys.exit(1 if e else 0)
