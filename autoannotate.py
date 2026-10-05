"""Auto-annotation: a detector PROPOSES boxes, a person ACCEPTS them.

Rule kept from the rest of the system (see the classifier's suggestions.json): a model-made label never
enters training until a person accepts it. Proposals are stored under the image entry's own
`"proposals"` key in annotations.json, NOT in `"boxes"`. The exporters (annotate.export_yolo_*) read only
`boxes`, so an unreviewed proposal can never reach a training set. Accepting moves a proposal into
`boxes` tagged `"source": "auto"`; the image stays `reviewed: false` until a person marks it reviewed.

This module is pure data (no Tk, no model import). `run()` takes any callable
`detect_fn(bgr) -> iterable of (class_name, confidence, x1, y1, x2, y2)` in absolute pixels, so the
Stage 2 YOLO, a future segmenter, or a fake in the self-test all plug in the same way.

Active learning (all three only reorder UNREVIEWED images; nothing is labelled by them):
  * `order_for_review`   least sure first (lowest proposal confidence, or no proposal at all);
  * `missing_part_queue` the model found the anchor (bottle) but not a required part (cap / label):
                         the most likely real missing-cap / missing-label captures, rare and valuable;
  * `doubtful_part_queue` a part proposed at middling confidence (default 0.25-0.80): hard-negative
                         candidates, e.g. the green tamper ring of a bare neck proposed as "cap" at 0.69.

    python autoannotate.py        # self-test
"""
from __future__ import annotations

import copy
from pathlib import Path

import annotate as A


def to_proposals(dets, img_wh, classes, min_conf: float = 0.25) -> list:
    """Absolute-pixel detections -> normalised YOLO-style proposal dicts, clipped to the image.
    Classes the project does not have are dropped, as are boxes below `min_conf` or with no area."""
    w, h = img_wh
    out = []
    for name, conf, x1, y1, x2, y2 in dets:
        if name not in classes or conf < min_conf:
            continue
        x1, x2 = sorted((min(max(float(x1), 0.0), w), min(max(float(x2), 0.0), w)))
        y1, y2 = sorted((min(max(float(y1), 0.0), h), min(max(float(y2), 0.0), h)))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        out.append({"cls": name, "x": round((x1 + x2) / 2 / w, 6), "y": round((y1 + y2) / 2 / h, 6),
                    "w": round(bw, 6), "h": round(bh, 6), "conf": round(float(conf), 4)})
    return out


def propose(data: dict, rel: str, dets, img_wh, min_conf: float = 0.25, overwrite: bool = False) -> int:
    """Store proposals for one image. Never touches `boxes` or `reviewed`. Returns how many were stored.
    An image a person already reviewed is left alone unless overwrite=True."""
    entry = data["images"].get(rel) or {"reviewed": False, "boxes": [], "polygons": []}
    if entry.get("reviewed") and not overwrite:
        return 0
    props = to_proposals(dets, img_wh, data["classes"], min_conf)
    for p in props:
        A._validate_box(p, data["classes"])
    data["images"][rel] = {**entry, "boxes": entry.get("boxes", []), "polygons": entry.get("polygons", []),
                           "reviewed": bool(entry.get("reviewed")), "proposals": props}
    return len(props)


def accept(data: dict, rel: str, indices=None) -> int:
    """Move proposals (all, or those at `indices`) into `boxes` as source=auto. The rest stay proposals."""
    entry = data["images"].get(rel)
    props = (entry or {}).get("proposals") or []
    pick = set(range(len(props))) if indices is None else set(indices)
    moved = [dict(p, source="auto") for i, p in enumerate(props) if i in pick]
    if not moved:
        return 0
    entry["boxes"] = list(entry.get("boxes", [])) + moved
    entry["proposals"] = [p for i, p in enumerate(props) if i not in pick]
    return len(moved)


def reject(data: dict, rel: str, indices=None) -> int:
    entry = data["images"].get(rel)
    props = (entry or {}).get("proposals") or []
    drop = set(range(len(props))) if indices is None else set(indices)
    entry["proposals"] = [p for i, p in enumerate(props) if i not in drop] if entry else []
    return len(drop & set(range(len(props))))


def pending_proposals(data: dict, rel: str) -> list:
    return list((data["images"].get(rel) or {}).get("proposals") or [])


def uncertainty(entry: dict | None) -> float:
    """0 = sure, 1 = very unsure. Lowest proposal confidence; an image with no proposal is maximally
    unsure (the model saw nothing -- exactly the image a person should look at)."""
    props = (entry or {}).get("proposals") or []
    return 1.0 - min((p.get("conf", 0.0) for p in props), default=0.0)


def order_for_review(data: dict, images: list) -> list:
    """Unreviewed images first, least confident first; reviewed images last, in their given order."""
    todo = [r for r in images if not (data["images"].get(r) or {}).get("reviewed")]
    done = [r for r in images if (data["images"].get(r) or {}).get("reviewed")]
    return sorted(todo, key=lambda r: -uncertainty(data["images"].get(r))) + done


def _props(data, r):
    return (data["images"].get(r) or {}).get("proposals") or []


def _todo(data, images):
    return [r for r in images if not (data["images"].get(r) or {}).get("reviewed")]


def missing_part_queue(data: dict, images: list, anchor: str = "bottle", parts=("cap", "label")) -> list:
    """Unreviewed images whose proposals contain the anchor but lack a part; most confident anchor first."""
    out = []
    for r in _todo(data, images):
        ps = _props(data, r)
        anc = [p.get("conf", 0.0) for p in ps if p.get("cls") == anchor]
        have = {p.get("cls") for p in ps}
        if anc and any(x not in have for x in parts):
            out.append((max(anc), r))
    return [r for _, r in sorted(out, key=lambda t: -t[0])]


def doubtful_part_queue(data: dict, images: list, parts=("cap", "label"), lo: float = 0.25, hi: float = 0.80) -> list:
    """Unreviewed images with a part proposed at lo <= conf < hi; closest to the middle of the band first."""
    mid = (lo + hi) / 2
    out = []
    for r in _todo(data, images):
        cs = [p.get("conf", 0.0) for p in _props(data, r) if p.get("cls") in parts and lo <= p.get("conf", 0.0) < hi]
        if cs:
            out.append((min(abs(c - mid) for c in cs), r))
    return [r for _, r in sorted(out)]


def run(data: dict, images: list, image_root: Path, detect_fn, imread, min_conf: float = 0.25,
        only_pending: bool = True, progress=None) -> dict:
    """Propose boxes for `images`. only_pending skips images that already have boxes or were reviewed."""
    stats = {"images": 0, "proposed": 0, "skipped": 0, "unreadable": 0, "empty": 0}
    for i, rel in enumerate(images):
        entry = data["images"].get(rel) or {}
        if only_pending and (entry.get("reviewed") or entry.get("boxes") or entry.get("polygons")):
            stats["skipped"] += 1
            continue
        img = imread(Path(image_root) / rel)
        if img is None:
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        n = propose(data, rel, list(detect_fn(img)), (w, h), min_conf)
        stats["images"] += 1
        stats["proposed"] += n
        stats["empty"] += n == 0
        if progress:
            progress(i + 1, len(images))
    return stats


def yolo_detect_fn(detector):
    """Adapt detect.YoloDetector to run()'s detect_fn."""
    def fn(bgr):
        res = detector.detect(bgr)
        return [(d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2) for d in res.detections]
    return fn


def demo():
    import numpy as np
    classes = ["bottle", "cap", "label"]
    data = A.init_annotations("detection", classes)
    imgs = {"a.jpg": np.zeros((200, 100, 3), np.uint8), "b.jpg": np.zeros((200, 100, 3), np.uint8),
            "c.jpg": np.zeros((200, 100, 3), np.uint8), "d.jpg": np.zeros((200, 100, 3), np.uint8)}
    fake = {"a.jpg": [("bottle", 0.95, 10, 10, 90, 190), ("cap", 0.9, 30, 5, 70, 30)],
            "b.jpg": [("bottle", 0.40, 10, 10, 90, 190), ("cup", 0.99, 0, 0, 50, 50)],   # 'cup' not a class
            "c.jpg": [],
            "d.jpg": [("label", 0.8, -20, 50, 130, 120)]}                               # clipped to the image
    cur = {}
    imread = lambda p: (cur.__setitem__("k", p.name), imgs[p.name])[1]                   # noqa: E731
    detect_fn = lambda img: fake[cur["k"]]                                               # noqa: E731
    A.set_image_annotation(data, "d.jpg", boxes=[], reviewed=True)                       # a person already did d
    st = run(data, list(imgs), Path("."), detect_fn, imread)
    assert st["images"] == 3 and st["skipped"] == 1 and st["proposed"] == 3 and st["empty"] == 1, st
    assert data["images"]["d.jpg"].get("proposals") is None, "a reviewed image was overwritten"
    # proposals are NOT annotations: they are not in boxes, and the exporter does not see them
    assert all(not data["images"][r]["boxes"] for r in ("a.jpg", "b.jpg", "c.jpg"))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        A.export_yolo_detection(data, Path(td))
        assert all(not f.read_text().strip() for f in Path(td).rglob("*.txt")), "a proposal was exported"
    A.validate_annotations(data)
    # accept one, reject one, the rest stay
    assert accept(data, "a.jpg", [0]) == 1
    e = data["images"]["a.jpg"]
    assert len(e["boxes"]) == 1 and e["boxes"][0]["source"] == "auto" and not e["reviewed"]
    assert len(e["proposals"]) == 1 and reject(data, "a.jpg") == 1 and e["proposals"] == []
    A.validate_annotations(data)
    # active learning: image with no proposal first, then lowest confidence
    order = order_for_review(data, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    assert set(order[:2]) == {"a.jpg", "c.jpg"} and order[2:] == ["b.jpg", "d.jpg"], order   # a,c: no proposal left
    # active-learning queues on fresh proposals
    q = A.init_annotations("detection", classes)
    for r, ps in (("full.jpg", [("bottle", 0.9, 10, 10, 90, 190), ("cap", 0.95, 30, 5, 70, 30), ("label", 0.9, 15, 80, 85, 140)]),
                  ("nocap.jpg", [("bottle", 0.85, 10, 10, 90, 190), ("label", 0.9, 15, 80, 85, 140)]),
                  ("neck.jpg", [("bottle", 0.82, 10, 10, 90, 190), ("cap", 0.69, 30, 5, 70, 30), ("label", 0.9, 15, 80, 85, 140)]),
                  ("empty.jpg", [])):
        propose(q, r, ps, (100, 200))
    imgs2 = ["full.jpg", "nocap.jpg", "neck.jpg", "empty.jpg"]
    assert missing_part_queue(q, imgs2) == ["nocap.jpg"], missing_part_queue(q, imgs2)
    assert doubtful_part_queue(q, imgs2) == ["neck.jpg"], doubtful_part_queue(q, imgs2)
    A.set_image_annotation(q, "nocap.jpg", boxes=[], reviewed=True)
    assert missing_part_queue(q, imgs2) == []                                           # reviewed: out of the queue
    # clipping and normalisation
    p = to_proposals([("label", 0.8, -20, 50, 130, 120)], (100, 200), classes)[0]
    assert p["x"] == 0.5 and p["w"] == 1.0 and abs(p["h"] - 0.35) < 1e-6, p
    assert to_proposals([("cap", 0.1, 0, 0, 10, 10)], (100, 200), classes) == []         # below min_conf
    assert to_proposals([("cap", 0.9, 10, 10, 10, 50)], (100, 200), classes) == []       # zero width
    print("ok  autoannotate: proposals kept out of boxes/export, accept/reject, reviewed images untouched, "
          "uncertainty ordering, missing-part and doubtful-part queues, clipping")


if __name__ == "__main__":
    demo()
