"""Stage 2 component detector runtime: YOLOv8n finds the bottle, cap and label.

What this is -- and is not
    It finds COMPONENTS (bottle / cap / label). It does not detect defects. A missing
    box is NOT a defect verdict: absence can be an occluded part, a bad angle, blur,
    lighting, an object outside the frame, low confidence, or a detector failure. No
    rule here turns a missing component into REJECT -- that is the later decision
    engine's job, and it needs more than one frame to make it responsibly.

Coordinate system (the one convention used everywhere in this module)
    Detection boxes are ABSOLUTE PIXELS IN THE ORIGINAL FRAME that was passed to
    detect(): origin top-left, x to the right, y down, (x1, y1) the top-left corner
    and (x2, y2) the bottom-right, 0 <= x <= frame width, 0 <= y <= frame height.
    Ultralytics letterboxes the frame to `imgsz` internally and maps its boxes back
    to the original image, so there is NO ROI/crop and no resize in this contract.
    (The Stage 1 classifier's ROI crop is a different thing and is not used here: the
    detector was trained on whole 1780x1000 frames.) scale_box() converts to another
    frame size, e.g. for drawing on a resized preview.

Development defaults, not production values
    DEV_CONF is a development confidence threshold. It has not been validated against
    the production camera, lighting or line.

    python detect.py        # self-test (fakes always; real model too if the weights exist)
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "stage2_yolo" / "stage2_best.pt"
PROVENANCE = ROOT / "models" / "stage2_yolo" / "MODEL_PROVENANCE.json"

CLASS_NAMES = ("bottle", "cap", "label")        # the bottle line's classes (stage2_best.pt); another
                                                # product's model brings its own -- names are read from
                                                # the model, and it must have the classes its recipe uses
DEV_CONF = 0.25                                 # DEVELOPMENT threshold -- not validated for production
DEV_IMGSZ = 640                                 # the size the model was trained and evaluated at


class DetectorError(Exception):
    """The detector cannot be trusted for this frame (or at all). Callers treat it as FAULT."""


class Detection(NamedTuple):
    """One box. Coordinates: absolute pixels in the original frame (see module docstring)."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "class_name": self.class_name,
                "confidence": round(self.confidence, 4),
                "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1)}


class DetectionResult(NamedTuple):
    """All detections for one frame, with where they came from."""
    camera_id: str
    frame_seq: int | None               # Camera's per-session sequence number of the frame
    frame_ts: float | None              # that frame's time.monotonic() stamp
    detections: tuple                   # tuple[Detection, ...], highest confidence first
    frame_wh: tuple                     # (width, height) of the frame the boxes refer to
    infer_ms: float                     # detect() duration, perf_counter clock
    model_id: str
    conf_threshold: float               # the DEVELOPMENT threshold in force

    def counts(self) -> dict:
        names = list(CLASS_NAMES) + sorted({d.class_name for d in self.detections} - set(CLASS_NAMES))
        return {n: sum(1 for d in self.detections if d.class_name == n) for n in names}


def scale_box(box, src_wh, dst_wh) -> tuple:
    """Map (x1, y1, x2, y2) from a frame of size src_wh to one of size dst_wh (plain scaling,
    no letterbox). Use for drawing boxes on a resized preview."""
    (sw, sh), (dw, dh) = src_wh, dst_wh
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        raise ValueError(f"bad frame sizes {src_wh} -> {dst_wh}")
    fx, fy = dw / sw, dh / sh
    x1, y1, x2, y2 = box
    return (x1 * fx, y1 * fy, x2 * fx, y2 * fy)


def parse_boxes(xyxy, conf, cls, frame_wh, names=CLASS_NAMES) -> tuple:
    """Raw model output -> validated Detections. Anything unusable raises DetectorError;
    nothing is silently dropped or repaired, because a quietly wrong box is worse than FAULT.

    Zero boxes is a VALID result (an empty frame, or a bottle not yet in view), not an error.
    """
    xyxy, conf, cls = (np.asarray(a, dtype=np.float64) for a in (xyxy, conf, cls))
    if xyxy.size == 0 and conf.size == 0 and cls.size == 0:
        return ()
    if xyxy.ndim != 2 or xyxy.shape[1] != 4 or conf.shape != (len(xyxy),) or cls.shape != (len(xyxy),):
        raise DetectorError(f"malformed detector output: xyxy{xyxy.shape} conf{conf.shape} cls{cls.shape}")
    if not (np.isfinite(xyxy).all() and np.isfinite(conf).all() and np.isfinite(cls).all()):
        raise DetectorError("non-finite value in detector output (box, confidence or class)")
    if (conf < 0).any() or (conf > 1).any():
        raise DetectorError(f"confidence outside [0, 1]: min {conf.min():.3g} max {conf.max():.3g}")
    if (cls != np.round(cls)).any() or (cls < 0).any() or (cls >= len(names)).any():
        raise DetectorError(f"class id outside 0..{len(names) - 1}: {sorted(set(cls.tolist()))}")
    w, h = frame_wh
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        x1, x2 = min(max(x1, 0.0), w), min(max(x2, 0.0), w)      # clip to the frame
        y1, y2 = min(max(y1, 0.0), h), min(max(y2, 0.0), h)
        if x2 <= x1 or y2 <= y1:
            raise DetectorError(f"degenerate box ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) in a {w}x{h} frame")
        out.append(Detection(int(k), names[int(k)], float(c), float(x1), float(y1), float(x2), float(y2)))
    out.sort(key=lambda d: -d.confidence)
    return tuple(out)


def verify_checkpoint(path, provenance=None) -> str:
    """sha256 of the weights, checked against MODEL_PROVENANCE.json. Refuses a file that is
    not the model we trained -- never substitute downloaded weights and call them ours."""
    path = Path(path)
    if provenance is None:                       # a product's detector carries its own provenance file
        own = path.parent / PROVENANCE.name
        provenance = own if own.is_file() else PROVENANCE
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        want = json.loads(Path(provenance).read_text(encoding="utf-8"))["model"]["checkpoint"]["sha256"]
    except Exception as e:                                       # noqa: BLE001
        raise DetectorError(f"cannot read the expected checksum from {provenance}: {e}") from e
    if got != want:
        raise DetectorError(f"{path.name} does not match the trained model recorded in "
                            f"{Path(provenance).name} (sha256 {got[:12]}... != {want[:12]}...)")
    return got


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class YoloDetector:
    """The Stage 2 YOLOv8n checkpoint behind one method: detect(frame) -> DetectionResult.

    One instance can be shared by several cameras: Ultralytics predictors keep state, so
    calls are serialised with a lock (correct first; the GPU is the bottleneck anyway).
    """

    def __init__(self, weights=None, conf: float = DEV_CONF, imgsz: int = DEV_IMGSZ, device=None,
                 verify: bool = True, warmup: bool = True, model=None, require=CLASS_NAMES):
        conf = float(conf)
        if not (0.0 < conf < 1.0):
            raise DetectorError(f"confidence threshold must be in (0, 1), got {conf}")
        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        self.conf, self.imgsz = conf, int(imgsz)
        self._lock = threading.Lock()
        self.model_id = "injected-model"
        if model is None:
            if not self.weights.is_file():
                raise DetectorError(f"detector weights not found: {self.weights}. Not downloading "
                                    f"substitutes: the trained checkpoint must be copied here "
                                    f"(see models/stage2_yolo/MODEL_PROVENANCE.json).")
            sha = verify_checkpoint(self.weights) if verify else hashlib.sha256(
                self.weights.read_bytes()).hexdigest()
            self.model_id = f"{self.weights.stem}@{sha[:12]}"
            try:
                from ultralytics import YOLO
                model = YOLO(str(self.weights))
            except Exception as e:                               # noqa: BLE001
                raise DetectorError(f"cannot load detector {self.weights.name}: "
                                    f"{type(e).__name__}: {e}") from e
        self._model = model
        names = getattr(model, "names", None)
        got = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names or ())
        lack = [n for n in require if n not in got]
        if not got or lack:
            raise DetectorError(f"model classes {got} lack {lack or list(require)}, which the inspection "
                                f"recipe needs: this is not the detector for this product")
        self.class_names = got
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:                                    # noqa: BLE001
                device = "cpu"
        self.device = device
        if warmup and self.weights.is_file():
            self.detect(np.zeros((self.imgsz * 9 // 16, self.imgsz, 3), np.uint8))   # pay CUDA init now

    def detect(self, frame, *, camera_id: str = "", frame_seq=None, frame_ts=None) -> DetectionResult:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or frame.size == 0):
            raise DetectorError("frame must be a non-empty HxWx3 uint8 BGR array")
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        try:
            with self._lock:
                res = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                          device=self.device, verbose=False)
        except Exception as e:                                   # noqa: BLE001
            raise DetectorError(f"inference failed: {type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        if not res or len(res) != 1:
            raise DetectorError(f"unexpected detector result ({len(res) if res else 0} images for 1)")
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            dets = ()
        else:
            dets = parse_boxes(_np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls), (w, h), self.class_names)
        return DetectionResult(camera_id, frame_seq, frame_ts, dets, (w, h), ms,
                               self.model_id, self.conf)


# ------------------------------------------------------------------------------ test doubles

class FakeYolo:
    """Stands in for ultralytics.YOLO in tests. `outputs(frame)` returns (xyxy, conf, cls)."""

    names = {0: "bottle", 1: "cap", 2: "label"}

    def __init__(self, outputs=None, delay: float = 0.0):
        self.outputs = outputs or (lambda f: ([[10, 20, 110, 220], [30, 20, 60, 50]], [0.9, 0.8], [0, 1]))
        self.delay, self.calls, self.active, self.max_active = delay, 0, 0, 0

    def predict(self, frame, **kw):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            xyxy, conf, cls = self.outputs(frame)
        finally:
            self.active -= 1

        class _B:
            def __init__(s):
                s.xyxy, s.conf, s.cls = (np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float),
                                         np.asarray(cls, float))
            def __len__(s):
                return len(s.conf)

        class _R:
            boxes = _B()
        return [_R()]


# ------------------------------------------------------------------------------ self-test

def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _expect(exc, fn, what):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what}: expected {exc.__name__}")


def demo(real: bool = True):
    import tempfile
    frame = np.zeros((1000, 1780, 3), np.uint8)
    wh = (1780, 1000)

    # valid parsing, class mapping, ordering, frame-coordinate boxes
    ds = parse_boxes([[10, 20, 110, 220], [30, 20, 60, 50], [500, 40, 700, 400]],
                     [0.5, 0.91, 0.7], [2, 1, 0], wh)
    assert [(d.class_name, d.class_id) for d in ds] == [("cap", 1), ("bottle", 0), ("label", 2)], ds
    assert ds[0].confidence == 0.91 and (ds[0].x1, ds[0].y1, ds[0].x2, ds[0].y2) == (30, 20, 60, 50)
    assert [CLASS_NAMES.index(n) for n in CLASS_NAMES] == [0, 1, 2]

    # empty result is VALID (not an error, not a defect)
    assert parse_boxes(np.zeros((0, 4)), [], [], wh) == ()
    det = YoloDetector(model=FakeYolo(lambda f: (np.zeros((0, 4)), [], [])), warmup=False)
    r = det.detect(frame, camera_id="0", frame_seq=7, frame_ts=1.5)
    assert r.detections == () and r.counts() == {"bottle": 0, "cap": 0, "label": 0}
    assert (r.camera_id, r.frame_seq, r.frame_ts, r.frame_wh) == ("0", 7, 1.5, wh)

    # invalid confidence / values -> DetectorError, never a quiet fix
    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        _expect(DetectorError, lambda b=bad: parse_boxes([[1, 1, 5, 5]], [b], [0], wh), f"conf {bad}")
    _expect(DetectorError, lambda: parse_boxes([[1, 1, float("nan"), 5]], [0.5], [0], wh), "nan box")
    _expect(DetectorError, lambda: parse_boxes([[5, 5, 5, 9]], [0.5], [0], wh), "degenerate box")
    _expect(DetectorError, lambda: parse_boxes([[1, 1, 5, 5]], [0.5, 0.4], [0], wh), "shape mismatch")
    for bad in (3, -1, 1.5):
        _expect(DetectorError, lambda b=bad: parse_boxes([[1, 1, 5, 5]], [0.5], [b], wh), f"class {bad}")

    # coordinates: clipped to the frame; conversion to another frame size
    c = parse_boxes([[-20, -5, 100, 5000]], [0.9], [0], wh)[0]
    assert (c.x1, c.y1, c.x2, c.y2) == (0, 0, 100, 1000), c
    assert scale_box((100, 50, 300, 250), (1780, 1000), (890, 500)) == (50.0, 25.0, 150.0, 125.0)
    _expect(ValueError, lambda: scale_box((0, 0, 1, 1), (0, 10), (5, 5)), "zero-size frame")

    # class names come from the model, so a different class order is read correctly, not mislabelled
    class Swapped(FakeYolo):
        names = {0: "cap", 1: "bottle", 2: "label"}
    sw = YoloDetector(model=Swapped(lambda f: ([[30, 20, 60, 50]], [0.9], [0])), warmup=False)
    assert [d.class_name for d in sw.detect(frame).detections] == ["cap"]
    # ...but a model without a class the recipe needs is refused
    class NoLabel(FakeYolo):
        names = {0: "bottle", 1: "cap"}
    _expect(DetectorError, lambda: YoloDetector(model=NoLabel(), warmup=False), "model lacks 'label'")
    class Boxes(FakeYolo):                       # another product: its own classes, its own recipe
        names = {0: "box", 1: "sticker"}
    bx = YoloDetector(model=Boxes(lambda f: ([[5, 5, 50, 50]], [0.9], [1])), warmup=False, require=("box", "sticker"))
    r = bx.detect(frame)
    assert [d.class_name for d in r.detections] == ["sticker"] and r.counts()["sticker"] == 1, r.counts()
    _expect(DetectorError, lambda: YoloDetector(model=Boxes(), warmup=False), "box model on the bottle recipe")

    # missing / non-matching weights
    _expect(DetectorError, lambda: YoloDetector(weights=Path("no/such/stage2_best.pt")), "missing weights")
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "w.pt"
        fake.write_bytes(b"not the trained model")
        _expect(DetectorError, lambda: YoloDetector(weights=fake), "foreign weights refused")
        try:                                             # ...and it must be the CHECKSUM that refuses them
            YoloDetector(weights=fake)
        except DetectorError as e:
            assert "does not match the trained model" in str(e), str(e)
        prov = Path(td) / "p.json"
        prov.write_text(json.dumps({"model": {"checkpoint": {"sha256": hashlib.sha256(
            b"not the trained model").hexdigest()}}}), encoding="utf-8")
        assert verify_checkpoint(fake, prov)
        prov.write_text(json.dumps({"model": {"checkpoint": {"sha256": "0" * 64}}}), encoding="utf-8")
        try:
            verify_checkpoint(fake, prov)
            raise AssertionError("a checkpoint with the wrong sha256 was accepted")
        except DetectorError as e:
            assert "does not match" in str(e), str(e)
        prov.write_text("{}", encoding="utf-8")
        _expect(DetectorError, lambda: verify_checkpoint(fake, prov), "unreadable provenance")
    _expect(DetectorError, lambda: YoloDetector(conf=0.0, model=FakeYolo(), warmup=False), "conf 0")
    _expect(DetectorError, lambda: YoloDetector(conf=1.0, model=FakeYolo(), warmup=False), "conf 1")

    # inference exception and bad frames -> DetectorError (the caller turns it into FAULT)
    def boom(f):
        raise RuntimeError("cuda out of memory")
    _expect(DetectorError, lambda: YoloDetector(model=FakeYolo(boom), warmup=False).detect(frame), "predict raises")
    d = YoloDetector(model=FakeYolo(), warmup=False)
    for bad in (None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((0, 10, 3), np.uint8)):
        _expect(DetectorError, lambda b=bad: d.detect(b), "bad frame")
    _expect(DetectorError, lambda: YoloDetector(model=FakeYolo(
        lambda f: ([[1, 1, 9, 9]], [float("nan")], [0])), warmup=False).detect(frame), "nan confidence end to end")

    # shared by several threads: calls never overlap
    fy = FakeYolo(delay=0.01)
    shared = YoloDetector(model=fy, warmup=False)
    ts = [threading.Thread(target=lambda: [shared.detect(frame) for _ in range(5)]) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert fy.calls == 20 and fy.max_active == 1, (fy.calls, fy.max_active)

    msg = "fakes: parsing, empty, invalid values, coordinates, class mapping, missing/foreign weights, errors, threads"
    msg += _real_model_check() if real else " | real model check skipped"
    print("ok  (" + msg + ")")


def _real_model_check() -> str:
    """Against the real checkpoint, if present: the same runtime path the camera uses must
    reproduce sane detections on Stage 2 validation frames (coordinate system included)."""
    import cv2
    val = ROOT / "stage2_dataset" / "yolo_export" / "images" / "val"
    lbl = ROOT / "stage2_dataset" / "yolo_export" / "labels" / "val"
    if not DEFAULT_WEIGHTS.is_file() or not val.is_dir():
        return " | real model: SKIPPED (weights or Stage 2 val images not present)"
    det = YoloDetector()                                         # verifies sha256 against provenance
    tp = fp = fn = 0
    files = sorted(val.glob("*.png"))[::6][:15]
    for f in files:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        gt = []
        for ln in (lbl / (f.stem + ".txt")).read_text().split("\n"):
            if ln.strip():
                k, cx, cy, bw, bh = ln.split()
                gt.append((int(k), (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
                           (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h))
        r = det.detect(im)
        assert r.frame_wh == (w, h) and all(0 <= d.x1 < d.x2 <= w and 0 <= d.y1 < d.y2 <= h for d in r.detections)
        used = set()
        for d in r.detections:
            m = [(i, _iou((d.x1, d.y1, d.x2, d.y2), g[1:])) for i, g in enumerate(gt)
                 if g[0] == d.class_id and i not in used]
            i, best = max(m, key=lambda t: t[1], default=(None, 0.0))
            if best >= 0.5:
                tp += 1; used.add(i)
            else:
                fp += 1
        fn += len(gt) - len(used)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    assert prec > 0.8 and rec > 0.8, f"real-model sanity failed: precision {prec:.2f} recall {rec:.2f}"
    return (f" | real model {det.model_id}: {len(files)} val frames, IoU>=0.5 same-class "
            f"precision {prec:.2f} recall {rec:.2f} @conf {det.conf} (sanity check, not an evaluation)")


if __name__ == "__main__":
    demo()
