"""Auto-annotation: a detector PROPOSES boxes, a person ACCEPTS them.

Rule kept from the rest of the system (see the classifier's suggestions.json): a model-made label never
enters training until a person accepts it. Proposals are stored under the image entry's own
`"proposals"` key in annotations.json, NOT in `"boxes"`. The exporters (annotate.export_yolo_*) read only
`boxes` (and skip an image that still holds undecided proposals), so an unreviewed proposal can never reach a
training set. Accepting moves a proposal into `boxes` tagged `"source": "auto"`; the image stays
`reviewed: false` until a person marks it reviewed.

Review vocabulary (per proposal). A proposal is PENDING under "proposals"; a person's decision moves it:
  ACCEPTED  -> "boxes", source "auto", the model's numbers kept (conf, model, decided)
  CORRECTED -> "boxes", source "auto", corrected=True, the model's original under "proposed"
  REJECTED  -> "rejected" (kept for the record and undo; never exported, hidden on the canvas by default)
Per image: "auto" = the last proposal run ({model, t, n}; n=0 is a valid empty result); "failed" = why that run
could not use the image; "deferred" = a person chose "skip for later". All of these are optional keys the schema
already tolerates: annotations.json stays schema_version 1 and older files load unchanged.

This module is pure data (no Tk, no model import). `detect_images()` takes any callable
`detect_fn(bgr) -> iterable of (class_name, confidence, x1, y1, x2, y2)` in absolute pixels, so the
Stage 2 YOLO, a future segmenter, or a fake in the self-test all plug in the same way.

Active learning (all three only reorder UNREVIEWED images; nothing is labelled by them, and an image in a queue
is a CANDIDATE for review, not a confirmed defect):
  * `order_for_review`   least sure first (lowest proposal confidence, or no proposal at all);
  * `missing_part_queue` the model found the anchor (bottle) but not a required part (cap / label):
                         likely real missing-cap / missing-label captures -- or occlusion / a poor angle;
  * `doubtful_part_queue` a part proposed at middling confidence (default 0.25-0.80): hard-negative
                         candidates, e.g. the green tamper ring of a bare neck proposed as "cap" at 0.69.

    python -m vision.autoannotate        # self-test
"""
from __future__ import annotations

import copy
import getpass
import json
import math
import time
from pathlib import Path

from vision import annotate as A

STATES = ("pending", "accepted", "corrected", "rejected", "failed", "skipped", "reviewed", "unreviewed",
          "not_proposed")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _finite01(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and 0.0 <= f <= 1.0


def _bump(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


# ------------------------------------------------------------------------------------------ proposals
def to_proposals(dets, img_wh, classes, min_conf: float = 0.25, stats: dict | None = None) -> list:
    """Absolute-pixel detections -> normalised YOLO-style proposal dicts, clipped to the image.
    Classes the project does not have are dropped (stats["unmapped"]), as are boxes below `min_conf`.
    A detection with a non-finite / out-of-range confidence, non-finite coordinates, no area or lying entirely
    outside the image is INVALID: dropped and counted in stats["invalid"], never silently turned into a box."""
    w, h = img_wh
    out = []
    for det in dets:
        try:
            name, conf, x1, y1, x2, y2 = det
            coords = [float(v) for v in (x1, y1, x2, y2)]
        except (TypeError, ValueError):
            _bump(stats, "invalid")
            continue
        if not _finite01(conf) or not all(math.isfinite(v) for v in coords):
            _bump(stats, "invalid")
            continue
        if name not in classes:
            _bump(stats, "unmapped")
            continue
        if float(conf) < min_conf:
            continue
        x1, y1, x2, y2 = coords
        x1, x2 = sorted((min(max(x1, 0.0), w), min(max(x2, 0.0), w)))
        y1, y2 = sorted((min(max(y1, 0.0), h), min(max(y2, 0.0), h)))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            _bump(stats, "invalid")
            continue
        out.append({"cls": name, "x": round((x1 + x2) / 2 / w, 6), "y": round((y1 + y2) / 2 / h, 6),
                    "w": round(bw, 6), "h": round(bh, 6), "conf": round(float(conf), 4)})
    return out


def propose(data: dict, rel: str, dets, img_wh, min_conf: float = 0.25, overwrite: bool = False,
            model: str | None = None, stats: dict | None = None) -> int:
    """Store proposals for one image. Never touches `boxes` or `reviewed`. Returns how many were stored.
    An image a person already reviewed is left alone unless overwrite=True. Zero detections is a VALID result:
    the image records the run ("auto", n=0), which is not the same as "never proposed" or "failed"."""
    entry = data["images"].get(rel) or {"reviewed": False, "boxes": [], "polygons": []}
    if entry.get("reviewed") and not overwrite:
        return 0
    props = to_proposals(dets, img_wh, data["classes"], min_conf, stats)
    for p in props:
        A._validate_box(p, data["classes"])
        if model:
            p["model"] = model
    new = {**entry, "boxes": entry.get("boxes", []), "polygons": entry.get("polygons", []),
           "reviewed": bool(entry.get("reviewed")), "proposals": props,
           "auto": {"model": model, "t": _now(), "n": len(props)}}
    new.pop("failed", None)
    data["images"][rel] = new
    return len(props)


def mark_failed(data: dict, rel: str, reason: str, model: str | None = None) -> None:
    """The run could not use this image (unreadable, inference error). Its proposals / boxes are untouched."""
    entry = data["images"].get(rel) or {"reviewed": False, "boxes": [], "polygons": []}
    data["images"][rel] = {**entry, "boxes": entry.get("boxes", []), "polygons": entry.get("polygons", []),
                           "failed": {"reason": str(reason)[:300], "model": model, "t": _now()}}


# ------------------------------------------------------------------------------------------ decisions
def accept(data: dict, rel: str, indices=None) -> int:
    """Move proposals (all, or those at `indices`) into `boxes` as source=auto. The rest stay proposals.
    The model's confidence and identity stay on the box; "decided" records when."""
    entry = data["images"].get(rel)
    props = (entry or {}).get("proposals") or []
    pick = set(range(len(props))) if indices is None else set(indices)
    t = _now()
    moved = [dict(p, source="auto", decided=t) for i, p in enumerate(props) if i in pick]
    if not moved:
        return 0
    entry["boxes"] = list(entry.get("boxes", [])) + moved
    entry["proposals"] = [p for i, p in enumerate(props) if i not in pick]
    entry.pop("deferred", None)
    return len(moved)


def reject(data: dict, rel: str, indices=None) -> int:
    """Move proposals to "rejected" (kept for the record and undo; never exported). Returns how many."""
    entry = data["images"].get(rel)
    if not entry:
        return 0
    props = entry.get("proposals") or []
    drop = set(range(len(props))) if indices is None else set(indices)
    t = _now()
    gone = [dict(p, decided=t) for i, p in enumerate(props) if i in drop]
    if not gone:
        return 0
    entry["rejected"] = list(entry.get("rejected") or []) + gone
    entry["proposals"] = [p for i, p in enumerate(props) if i not in drop]
    entry.pop("deferred", None)
    return len(gone)


def edit_proposal(data: dict, rel: str, index: int) -> int:
    """Start correcting a proposal: it becomes an accepted box that remembers the model's original under
    "proposed". Returns the new box's index. Every later change is judged by note_correction()."""
    entry = data["images"][rel]
    p = entry["proposals"].pop(index)
    orig = {k: p[k] for k in ("cls", "x", "y", "w", "h", "conf", "model") if k in p}
    box = {k: p[k] for k in ("cls", "x", "y", "w", "h")}
    box.update(source="auto", proposed=orig, corrected=False, decided=_now())
    for k in ("conf", "model"):
        if k in p:
            box[k] = p[k]
    entry["boxes"] = list(entry.get("boxes", [])) + [box]
    entry.pop("deferred", None)
    return len(entry["boxes"]) - 1


def note_correction(box: dict, tol: float = 1e-4) -> bool:
    """corrected = the person changed the class or the geometry of a box that came from a proposal."""
    orig = box.get("proposed")
    if not orig:
        return False
    changed = box["cls"] != orig.get("cls") or any(abs(float(box[k]) - float(orig.get(k, box[k]))) > tol
                                                   for k in ("x", "y", "w", "h"))
    box["corrected"] = bool(changed)
    if changed:
        box["decided"] = _now()
    return box["corrected"]


def skip(data: dict, rel: str) -> None:
    """Skip for later: the image keeps its proposals and is marked deferred (the Skipped filter finds it again)."""
    entry = data["images"].setdefault(rel, {"reviewed": False, "boxes": [], "polygons": []})
    entry["deferred"] = True


def mark_reviewed(data: dict, rel: str, reject_pending: bool = False) -> int:
    """A person vouches for this image's boxes (an empty image = a true background). Refused while proposals are
    still undecided, unless reject_pending (then they are rejected first). Returns how many were rejected."""
    entry = data["images"].setdefault(rel, {"reviewed": False, "boxes": [], "polygons": []})
    n = 0
    if entry.get("proposals"):
        if not reject_pending:
            raise A.AnnotationError(f"{rel}: {len(entry['proposals'])} proposals are still undecided")
        n = reject(data, rel)
    entry["reviewed"] = True
    entry.pop("deferred", None)
    return n


# ------------------------------------------------------------------------------------------ states / counts
def image_states(entry: dict | None) -> set:
    """Every state that describes this image (an image can be several: accepted AND rejected AND reviewed)."""
    e = entry or {}
    s = set()
    boxes = e.get("boxes") or []
    if e.get("proposals"):
        s.add("pending")
    if any(b.get("source") == "auto" and not b.get("corrected") for b in boxes):
        s.add("accepted")
    if any(b.get("corrected") for b in boxes):
        s.add("corrected")
    if e.get("rejected"):
        s.add("rejected")
    if e.get("failed"):
        s.add("failed")
    if e.get("deferred"):
        s.add("skipped")
    s.add("reviewed" if e.get("reviewed") else "unreviewed")
    if not e.get("auto") and not e.get("proposals") and not e.get("failed"):
        s.add("not_proposed")
    return s


def state_word(entry: dict | None) -> str:
    """The one word the image list shows (the most actionable state first)."""
    s = image_states(entry)
    for w in ("failed", "pending", "skipped", "corrected", "accepted", "rejected"):
        if w in s:
            return w
    if "reviewed" in s:
        return "reviewed"
    if (entry or {}).get("boxes") or (entry or {}).get("polygons"):
        return "annotated"
    return "not proposed" if "not_proposed" in s else "no objects"


def counts(data: dict, images: list) -> dict:
    """The page's metric cards, from what is stored (never estimated)."""
    c = {"images": len(images), "proposed_images": 0, "pending": 0, "pending_images": 0, "accepted": 0,
         "corrected": 0, "rejected": 0, "failed": 0, "reviewed": 0, "manual": 0}
    for r in images:
        e = data["images"].get(r) or {}
        c["proposed_images"] += bool(e.get("auto"))
        n = len(e.get("proposals") or [])
        c["pending"] += n
        c["pending_images"] += n > 0
        for b in e.get("boxes") or []:
            if b.get("corrected"):
                c["corrected"] += 1
            elif b.get("source") == "auto":
                c["accepted"] += 1
            else:
                c["manual"] += 1
        c["rejected"] += len(e.get("rejected") or [])
        c["failed"] += bool(e.get("failed"))
        c["reviewed"] += bool(e.get("reviewed"))
    decided = c["accepted"] + c["corrected"] + c["rejected"]
    c["decided_pct"] = round(100.0 * decided / (decided + c["pending"]), 1) if decided + c["pending"] else 0.0
    return c


def matches(entry: dict | None, state: str = "all", cls: str | None = None, lo: float = 0.0,
            hi: float = 1.0) -> bool:
    """Queue filter. `cls` / confidence range apply to the image's pending proposals (the thing being reviewed);
    for an image without proposals they apply to its boxes."""
    e = entry or {}
    if state not in ("all", None, "") and state not in image_states(e):
        return False
    if cls in (None, "", "all") and lo <= 0.0 and hi >= 1.0:
        return True
    items = e.get("proposals") or e.get("boxes") or []
    return any((cls in (None, "", "all") or b.get("cls") == cls) and lo <= float(b.get("conf", 1.0)) <= hi
               for b in items)


def select_proposals(data: dict, images: list, cls: str | None = None, lo: float = 0.0, hi: float = 1.0,
                     model: str | None = None) -> dict:
    """{rel: [proposal indices]} matching a bulk filter. Only PENDING proposals are ever selected: a bulk action
    can never touch an accepted, corrected, rejected or hand-drawn box."""
    out = {}
    for r in images:
        idx = [i for i, p in enumerate((data["images"].get(r) or {}).get("proposals") or [])
               if (cls in (None, "", "all") or p.get("cls") == cls) and lo <= float(p.get("conf", 0.0)) <= hi
               and (model in (None, "", "all") or p.get("model") == model)]
        if idx:
            out[r] = idx
    return out


def bulk(data: dict, selection: dict, action: str) -> int:
    """Apply accept / reject to exactly the selected proposals. Returns how many changed."""
    fn = {"accept": accept, "reject": reject}[action]
    return sum(fn(data, r, idx) for r, idx in selection.items())


def models_used(data: dict) -> list:
    return sorted({p.get("model") for e in data["images"].values() for p in (e.get("proposals") or [])
                   if p.get("model")})


def pending_proposals(data: dict, rel: str) -> list:
    return list((data["images"].get(rel) or {}).get("proposals") or [])


# ------------------------------------------------------------------------------------------ audit log
def log_action(path: Path, action: str, **detail) -> None:
    """Append one line to the annotation audit log (JSON lines, next to annotations.json). Never raises."""
    try:
        rec = {"time": _now(), "user": getpass.getuser(), "action": action, **detail}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:                                            # noqa: BLE001 - the log must never block work
        pass


def read_log(path: Path, n: int = 200) -> list:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------------------------------ active learning
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


def doubtful_part_queue(data: dict, images: list, parts=("cap", "label"), lo: float = 0.25,
                        hi: float = 0.80) -> list:
    """Unreviewed images with a part proposed at lo <= conf < hi; closest to the middle of the band first."""
    mid = (lo + hi) / 2
    out = []
    for r in _todo(data, images):
        cs = [p.get("conf", 0.0) for p in _props(data, r) if p.get("cls") in parts and lo <= p.get("conf", 0.0) < hi]
        if cs:
            out.append((min(abs(c - mid) for c in cs), r))
    return [r for _, r in sorted(out)]


def predict(entry: dict | None, img_wh, recipe=None, use: str = "auto") -> tuple:
    """What the production rule would say about this image, in words, from its boxes:
    ('GOOD' | 'DEFECT' | 'NO BOTTLE' | '', [defects], source). source: 'proposals' (the model's, unreviewed) or
    'boxes' (the person's). use='auto' prefers proposals when there are any. Uses decision.detection_findings with
    the project's recipe, so the Annotate page and the line judge a picture the same way."""
    from line import decision as DEC
    from vision import detect
    e = entry or {}
    props, boxes = e.get("proposals") or [], e.get("boxes") or []
    if use == "proposals" or (use == "auto" and props):
        items, source = props, "proposals"
    else:
        items, source = boxes, "boxes"
    if not items:
        return "", [], source
    w, h = img_wh
    dets = tuple(detect.Detection(-1, b["cls"], float(b.get("conf", 1.0)),
                                  (b["x"] - b["w"] / 2) * w, (b["y"] - b["h"] / 2) * h,
                                  (b["x"] + b["w"] / 2) * w, (b["y"] + b["h"] / 2) * h) for b in items)
    rules = dict(DEC.RULES)
    if recipe:
        rules["recipe"] = recipe
    found, _ = DEC.detection_findings(detect.DetectionResult("annotate", None, None, dets, (w, h), 0.0, "", 0.0), rules)
    if found is None:
        return "NO BOTTLE", [], source
    return ("DEFECT" if found else "GOOD"), list(found), source


# ------------------------------------------------------------------------------------------ running a model
def detect_images(images: list, image_root: Path, detect_fn, imread, should_stop=None, wait=None,
                  max_consecutive_failures: int = 5):
    """Generator, pure (no annotation data touched): one result per image, in order.
        ("ok", rel, (w, h), detections)   the detector answered (detections may be empty: a valid result)
        ("failed", rel, None, reason)     unreadable image or inference error -- that image only
    Stops when should_stop() is true (cancel), and after `max_consecutive_failures` failures in a row (a broken
    model or missing GPU fails every image: stop and say so, never retry forever) -> ("aborted", None, None, why).
    `wait` is called before each image (pause)."""
    fails = 0
    for rel in images:
        if wait is not None:
            wait()
        if should_stop is not None and should_stop():
            return
        img = imread(Path(image_root) / rel)
        if img is None or getattr(img, "size", 0) == 0:
            fails += 1
            yield ("failed", rel, None, "unreadable image")
        else:
            h, w = img.shape[:2]
            try:
                dets = list(detect_fn(img))
                fails = 0
                yield ("ok", rel, (w, h), dets)
            except Exception as e:                                # noqa: BLE001 - recorded per image
                fails += 1
                yield ("failed", rel, None, f"inference failed: {type(e).__name__}: {e}")
        if fails >= max_consecutive_failures:
            yield ("aborted", None, None, f"{fails} images in a row failed - run stopped")
            return


def apply_result(data: dict, result, min_conf: float = 0.25, model: str | None = None, stats: dict | None = None,
                 overwrite: bool = False) -> None:
    """Merge one detect_images() result into the annotation data (done by whoever owns `data`)."""
    kind, rel, wh, payload = result
    st = stats if stats is not None else {}
    if kind == "ok":
        entry = data["images"].get(rel) or {}
        if entry.get("reviewed") and not overwrite:
            _bump(st, "skipped")
            return
        n = propose(data, rel, payload, wh, min_conf, overwrite=overwrite, model=model, stats=st)
        _bump(st, "images")
        st["proposed"] = st.get("proposed", 0) + n
        st["empty"] = st.get("empty", 0) + (n == 0)
    elif kind == "failed":
        mark_failed(data, rel, payload, model)
        _bump(st, "failed")
        if payload == "unreadable image":
            _bump(st, "unreadable")


def run(data: dict, images: list, image_root: Path, detect_fn, imread, min_conf: float = 0.25,
        only_pending: bool = True, progress=None, model: str | None = None) -> dict:
    """Propose boxes for `images` (in-place, synchronous). only_pending skips images that already have boxes or
    were reviewed."""
    stats = {"images": 0, "proposed": 0, "skipped": 0, "unreadable": 0, "empty": 0, "failed": 0, "invalid": 0}
    todo = []
    for rel in images:
        entry = data["images"].get(rel) or {}
        if only_pending and (entry.get("reviewed") or entry.get("boxes") or entry.get("polygons")):
            stats["skipped"] += 1
        else:
            todo.append(rel)
    for i, res in enumerate(detect_images(todo, image_root, detect_fn, imread)):
        if res[0] == "aborted":
            stats["aborted"] = res[3]
            break
        apply_result(data, res, min_conf, model, stats)
        if progress:
            progress(i + 1, len(todo))
    return stats


def yolo_detect_fn(detector):
    """Adapt detect.YoloDetector to detect_images()' detect_fn."""
    def fn(bgr):
        res = detector.detect(bgr)
        return [(d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2) for d in res.detections]
    return fn


def detector_choices(root: Path, settings: dict | None = None) -> list:
    """[(label, path)] of detector weights on disk that could propose boxes: the configured one first, then the
    default checkpoint, then every candidate run (models/stage2_yolo/*/weights/best.pt). Missing files are left
    out -- a choice is never substituted by another checkpoint."""
    root = Path(root)
    seen, out = set(), []

    def add(label, p):
        p = Path(p)
        p = p if p.is_absolute() else root / p
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append((label, p))

    w = (settings or {}).get("detector_weights")
    if w:
        add(f"configured: {Path(w).name}", w)
    add("stage2_best.pt (default)", root / "models" / "stage2_yolo" / "stage2_best.pt")
    for p in sorted((root / "models" / "stage2_yolo").glob("*/weights/best.pt")):
        add(f"{p.parent.parent.name} (candidate)", p)
    return out


def class_mapping(model_classes, project_classes) -> tuple:
    """(mapped, unmapped_model_classes, project_classes_never_proposed). No overlap = incompatible model."""
    mc, pc = list(model_classes), list(project_classes)
    return ([c for c in mc if c in pc], [c for c in mc if c not in pc], [c for c in pc if c not in mc])


# ------------------------------------------------------------------------------------------ self-test
def demo():
    import tempfile

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
    st = run(data, list(imgs), Path("."), detect_fn, imread, model="fake@1")
    assert st["images"] == 3 and st["skipped"] == 1 and st["proposed"] == 3 and st["empty"] == 1, st
    assert st.get("unmapped") == 1, st
    assert data["images"]["d.jpg"].get("proposals") is None, "a reviewed image was overwritten"
    # zero detections is a VALID empty result, recorded as a run -- not "never proposed", not "failed"
    c_e = data["images"]["c.jpg"]
    assert c_e["auto"]["n"] == 0 and "failed" not in c_e and "not_proposed" not in image_states(c_e)
    assert data["images"]["a.jpg"]["proposals"][0]["model"] == "fake@1"
    # proposals are NOT annotations: they are not in boxes, and the exporter does not see them
    assert all(not data["images"][r]["boxes"] for r in ("a.jpg", "b.jpg", "c.jpg"))
    with tempfile.TemporaryDirectory() as td:
        A.export_yolo_detection(data, Path(td))
        assert all(not f.read_text().strip() for f in Path(td).rglob("*.txt")), "a proposal was exported"
    A.validate_annotations(data)
    # accept one, reject one, the rest stay
    assert accept(data, "a.jpg", [0]) == 1
    e = data["images"]["a.jpg"]
    assert len(e["boxes"]) == 1 and e["boxes"][0]["source"] == "auto" and not e["reviewed"]
    assert e["boxes"][0]["conf"] == 0.95 and e["boxes"][0]["model"] == "fake@1" and e["boxes"][0]["decided"]
    # partial decision: an image with an undecided proposal is NOT exported (its label file would teach the
    # pending cap as background)
    with tempfile.TemporaryDirectory() as td:
        assert "a.jpg" not in A.export_yolo_detection(data, Path(td))
    assert len(e["proposals"]) == 1 and reject(data, "a.jpg") == 1 and e["proposals"] == []
    assert len(e["rejected"]) == 1 and e["rejected"][0]["cls"] == "cap"           # kept for the record
    with tempfile.TemporaryDirectory() as td:
        assert sorted(A.export_yolo_detection(data, Path(td))) == ["a.jpg", "d.jpg"]
        assert (Path(td) / "a.txt").read_text().split()[0] == "0"                     # bottle only; cap rejected
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
    # the plain-words prediction the Annotate page shows (same rule as the line)
    pr = A.init_annotations("detection", classes)
    propose(pr, "ok.jpg", [("bottle", 0.9, 10, 10, 90, 190), ("cap", 0.9, 30, 5, 70, 30), ("label", 0.9, 15, 80, 85, 140)], (100, 200))
    propose(pr, "nc.jpg", [("bottle", 0.9, 10, 10, 90, 190), ("label", 0.9, 15, 80, 85, 140)], (100, 200))
    propose(pr, "nb.jpg", [("cap", 0.9, 30, 5, 70, 30)], (100, 200))
    assert predict(pr["images"]["ok.jpg"], (100, 200))[:2] == ("GOOD", [])
    assert predict(pr["images"]["nc.jpg"], (100, 200))[:2] == ("DEFECT", ["missing_cap"])
    assert predict(pr["images"]["nb.jpg"], (100, 200))[0] == "NO BOTTLE"
    assert predict({}, (100, 200))[0] == ""
    accept(pr, "nc.jpg")
    assert predict(pr["images"]["nc.jpg"], (100, 200))[2] == "boxes"
    # clipping and normalisation
    p = to_proposals([("label", 0.8, -20, 50, 130, 120)], (100, 200), classes)[0]
    assert p["x"] == 0.5 and p["w"] == 1.0 and abs(p["h"] - 0.35) < 1e-6, p
    assert to_proposals([("cap", 0.1, 0, 0, 10, 10)], (100, 200), classes) == []         # below min_conf
    assert to_proposals([("cap", 0.9, 10, 10, 10, 50)], (100, 200), classes) == []       # zero width
    print("ok  autoannotate: proposals kept out of boxes/export, accept/reject, reviewed images untouched, "
          "uncertainty ordering, missing-part and doubtful-part queues, plain-words prediction, clipping")

    # ---- invalid detections: NaN / out-of-range confidence, NaN coordinates, box outside the image, junk tuple
    st = {}
    out = to_proposals([("cap", float("nan"), 0, 0, 10, 10), ("cap", 1.7, 0, 0, 10, 10), ("cap", -0.1, 0, 0, 10, 10),
                        ("cap", 0.9, float("inf"), 0, 10, 10), ("cap", 0.9, 500, 500, 600, 600), ("bad",),
                        ("cap", 0.9, 10, 10, 40, 40)], (100, 200), classes, stats=st)
    assert len(out) == 1 and st["invalid"] == 6, (out, st)
    print("ok  invalid detections (NaN / out-of-range confidence, non-finite or outside boxes) dropped and counted")

    # ---- run loop: unreadable image and an inference error fail ONE image; the run goes on; a broken model aborts
    d2 = A.init_annotations("detection", classes)
    pics = {"ok1.jpg": np.zeros((50, 50, 3), np.uint8), "boom.jpg": np.zeros((50, 50, 3), np.uint8),
            "ok2.jpg": np.zeros((50, 50, 3), np.uint8)}
    seen = []

    def rd(p):
        seen.append(p.name)
        return pics.get(p.name)

    def det(img):
        if seen[-1] == "boom.jpg":
            raise RuntimeError("CUDA error")
        return [("bottle", 0.9, 5, 5, 45, 45)]
    st2 = run(d2, ["ok1.jpg", "missing.jpg", "boom.jpg", "ok2.jpg"], Path("."), det, rd, model="m")
    assert st2["images"] == 2 and st2["failed"] == 2 and st2["unreadable"] == 1 and "aborted" not in st2, st2
    assert "failed" in image_states(d2["images"]["missing.jpg"]) and "failed" in image_states(d2["images"]["boom.jpg"])
    assert "CUDA" in d2["images"]["boom.jpg"]["failed"]["reason"]
    A.validate_annotations(d2)
    # a re-run that succeeds clears the failure
    propose(d2, "boom.jpg", [("bottle", 0.9, 5, 5, 45, 45)], (50, 50), model="m")
    assert "failed" not in d2["images"]["boom.jpg"]
    many = ["x%d.jpg" % i for i in range(20)]
    res = list(detect_images(many, Path("."), det, lambda p: None))
    assert res[-1][0] == "aborted" and len(res) == 6, res                               # 5 failures, then stop
    # cancel: should_stop is honoured between images, completed results are kept by whoever applied them
    stop = {"n": 0}

    def should_stop():
        stop["n"] += 1
        return stop["n"] > 2
    got = list(detect_images(["ok1.jpg", "ok2.jpg", "ok1.jpg", "ok2.jpg"], Path("."), lambda i: [], rd, should_stop))
    assert len(got) == 2, got
    print("ok  run loop: unreadable / inference error fail one image only, broken model aborts after 5, "
          "cancel stops between images")

    # ---- edit / correct, skip, mark reviewed, states, counts
    d3 = A.init_annotations("detection", classes)
    propose(d3, "e.jpg", [("bottle", 0.9, 10, 10, 90, 190), ("cap", 0.6, 30, 5, 70, 30), ("label", 0.3, 15, 80, 85, 140)],
            (100, 200), model="m1")
    i = edit_proposal(d3, "e.jpg", 1)                       # the cap
    b = d3["images"]["e.jpg"]["boxes"][i]
    assert b["proposed"]["cls"] == "cap" and not note_correction(b)          # unchanged -> plain accept
    b["cls"], b["y"] = "label", b["y"] + 0.05
    assert note_correction(b) and b["corrected"]
    assert "corrected" in image_states(d3["images"]["e.jpg"]) and "pending" in image_states(d3["images"]["e.jpg"])
    try:
        mark_reviewed(d3, "e.jpg")
        raise AssertionError("marked reviewed with undecided proposals")
    except A.AnnotationError:
        pass
    skip(d3, "e.jpg")
    assert state_word(d3["images"]["e.jpg"]) == "pending" and "skipped" in image_states(d3["images"]["e.jpg"])
    assert mark_reviewed(d3, "e.jpg", reject_pending=True) == 2
    e3 = d3["images"]["e.jpg"]
    assert e3["reviewed"] and not e3["proposals"] and len(e3["rejected"]) == 2 and "deferred" not in e3
    A.validate_annotations(d3)
    with tempfile.TemporaryDirectory() as td:
        A.export_yolo_detection(d3, Path(td))
        line = (Path(td) / "e.txt").read_text().split()
        assert line[0] == "2" and abs(float(line[2]) - b["y"]) < 1e-6, line           # corrected class + geometry
    c = counts(d3, ["e.jpg", "none.jpg"])
    assert c["corrected"] == 1 and c["rejected"] == 2 and c["pending"] == 0 and c["decided_pct"] == 100.0, c
    print("ok  edit -> corrected keeps the model's original, skip, reviewed refused while proposals are undecided, "
          "corrected class + geometry exported, counts")

    # ---- bulk: only pending proposals matching the filter change; accepted / rejected / manual boxes untouched
    d4 = A.init_annotations("detection", classes)
    for r in ("p.jpg", "q.jpg"):
        propose(d4, r, [("bottle", 0.95, 10, 10, 90, 190), ("cap", 0.5, 30, 5, 70, 30)], (100, 200), model="m2")
    d4["images"]["q.jpg"]["boxes"].append({"cls": "label", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2})   # hand-drawn
    sel = select_proposals(d4, ["p.jpg", "q.jpg"], cls="bottle", lo=0.9)
    assert sel == {"p.jpg": [0], "q.jpg": [0]}, sel
    before = copy.deepcopy(d4)
    assert bulk(d4, sel, "accept") == 2
    for r in ("p.jpg", "q.jpg"):
        assert [p["cls"] for p in d4["images"][r]["proposals"]] == ["cap"]
    assert d4["images"]["q.jpg"]["boxes"][0] == before["images"]["q.jpg"]["boxes"][0]
    rej = select_proposals(d4, ["p.jpg", "q.jpg"], hi=0.6)
    assert bulk(d4, rej, "reject") == 2 and counts(d4, ["p.jpg", "q.jpg"])["accepted"] == 2
    assert select_proposals(d4, ["p.jpg"], model="other") == {}
    assert models_used(before) == ["m2"]
    assert matches(before["images"]["p.jpg"], "pending", cls="cap", lo=0.4, hi=0.6)
    assert not matches(before["images"]["p.jpg"], "pending", cls="cap", lo=0.7)
    assert not matches(d4["images"]["p.jpg"], "pending")
    print("ok  bulk accept / reject change only the selected pending proposals; hand-drawn and decided boxes untouched")

    # ---- audit log, detector choices, class mapping
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "annotation_log.jsonl"
        log_action(lp, "accept", image="a.jpg", n=1)
        log_action(lp, "bulk_reject", n=2)
        got = read_log(lp)
        assert [g["action"] for g in got] == ["accept", "bulk_reject"] and got[0]["user"]
        root = Path(td)
        (root / "models" / "stage2_yolo" / "run1" / "weights").mkdir(parents=True)
        (root / "models" / "stage2_yolo" / "run1" / "weights" / "best.pt").write_bytes(b"x")
        ch = detector_choices(root, {"detector_weights": "no/such.pt"})
        assert [c[0] for c in ch] == ["run1 (candidate)"], ch                          # missing files never listed
    m, um, never = class_mapping(["bottle", "cap", "label"], ["bottle", "cap", "label", "cap_tilted"])
    assert m == ["bottle", "cap", "label"] and um == [] and never == ["cap_tilted"]
    assert class_mapping(["person", "car"], classes)[0] == []                            # incompatible model
    print("ok  audit log, detector choices (no substitution), class mapping")


if __name__ == "__main__":
    demo()
