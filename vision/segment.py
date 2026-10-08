"""Segmentation runtime: pixel masks for bottle / cap / label, and the measurements a decision
rule can use (label area, label boundary regularity, component geometry).

Status: the RUNTIME INTERFACE exists; NO SEGMENTATION MODEL IS TRAINED YET. Annotation Studio can
draw polygons and export a YOLO-seg dataset; nothing has been annotated or trained. Until weights
exist at DEFAULT_WEIGHTS (or settings.json "segmenter_weights"), YoloSegmenter raises
SegmenterError and an inspection that requires segmentation is FAULT -- it never passes a bottle
it could not measure. The detection-based pipeline does not depend on this module.

Coordinate system: the same as detect.py -- absolute pixels in the original frame passed to
segment(), origin top-left. Polygons are (N, 2) float arrays of (x, y).

    python segment.py      # self-test with a fake model (no weights needed)
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "stage3_seg" / "seg_best.pt"
CLASS_NAMES = ("bottle", "cap", "label")     # same component classes as the detector
DEV_CONF = 0.25                              # DEVELOPMENT threshold, like detect.DEV_CONF
DEV_IMGSZ = 640


class SegmenterError(Exception):
    """The segmenter cannot be trusted for this frame (or at all). Callers treat it as FAULT."""


class Segment(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    polygon: np.ndarray                      # (N, 2) float, original-frame pixels
    area_px: float                           # polygon area
    box: tuple                               # (x1, y1, x2, y2) bounding box of the polygon

    @property
    def fill(self) -> float:
        """Polygon area / its bounding-box area: ~1 for an intact rectangular label, lower for a
        torn, folded or partly missing one."""
        x1, y1, x2, y2 = self.box
        return self.area_px / max(1e-6, (x2 - x1) * (y2 - y1))


class SegmentationResult(NamedTuple):
    camera_id: str
    frame_seq: int | None
    frame_ts: float | None
    segments: tuple                          # tuple[Segment, ...], highest confidence first
    frame_wh: tuple
    infer_ms: float
    model_id: str
    conf_threshold: float

    def best(self, name: str):
        return next((s for s in self.segments if s.class_name == name), None)

    def measurements(self) -> dict:
        """Numbers the decision engine reads. A value that cannot be computed is None (never 0):
        no bottle mask means no ratio, not a ratio of zero."""
        bottle, cap, label = self.best("bottle"), self.best("cap"), self.best("label")
        return {
            "bottle_area_px": bottle.area_px if bottle else None,
            "label_area_px": label.area_px if label else None,
            "label_area_ratio": (label.area_px / bottle.area_px) if (bottle and label and bottle.area_px > 0) else
                                (0.0 if bottle and not label else None),
            "label_fill": label.fill if label else None,
            "cap_present": cap is not None if bottle else None,
        }


def polygon_segment(class_id: int, conf: float, poly) -> Segment:
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3 or not np.isfinite(p).all():
        raise SegmenterError(f"degenerate polygon for class {class_id} ({len(p)} points)")
    area = float(abs(cv2.contourArea(p)))
    x1, y1 = float(p[:, 0].min()), float(p[:, 1].min())
    x2, y2 = float(p[:, 0].max()), float(p[:, 1].max())
    return Segment(int(class_id), CLASS_NAMES[int(class_id)], float(conf), p, area, (x1, y1, x2, y2))


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def available(weights=None) -> tuple:
    """(True, "") if a segmentation checkpoint exists, else (False, why). Cheap: no model load."""
    w = Path(weights) if weights else DEFAULT_WEIGHTS
    if not w.is_file():
        return False, f"no segmentation model trained (expected weights at {w})"
    return True, ""


class YoloSegmenter:
    """An Ultralytics YOLO-seg checkpoint behind one method: segment(frame) -> SegmentationResult.
    Calls are serialised (shared Ultralytics predictors keep state)."""

    def __init__(self, weights=None, conf: float = DEV_CONF, imgsz: int = DEV_IMGSZ, device=None, model=None):
        conf = float(conf)
        if not (0.0 < conf < 1.0):
            raise SegmenterError(f"confidence threshold must be in (0, 1), got {conf}")
        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        self.conf, self.imgsz = conf, int(imgsz)
        self._lock = threading.Lock()
        self.model_id = "injected-model"
        if model is None:
            ok, why = available(self.weights)
            if not ok:
                raise SegmenterError(why)
            sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
            self.model_id = f"{self.weights.stem}@{sha[:12]}"
            try:
                from ultralytics import YOLO
                model = YOLO(str(self.weights))
            except Exception as e:                           # noqa: BLE001
                raise SegmenterError(f"cannot load segmenter {self.weights.name}: {type(e).__name__}: {e}") from e
            if getattr(model, "task", "segment") != "segment":
                raise SegmenterError(f"{self.weights.name} is a '{model.task}' model, not a segmentation model")
        self._model = model
        names = getattr(model, "names", None)
        got = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names or ())
        if got != CLASS_NAMES:
            raise SegmenterError(f"model classes {got} != expected {CLASS_NAMES}")
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:                                # noqa: BLE001
                device = "cpu"
        self.device = device

    def segment(self, frame, *, camera_id: str = "", frame_seq=None, frame_ts=None) -> SegmentationResult:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or frame.size == 0):
            raise SegmenterError("frame must be a non-empty HxWx3 uint8 BGR array")
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        try:
            with self._lock:
                res = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        except Exception as e:                               # noqa: BLE001
            raise SegmenterError(f"inference failed: {type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        if not res or len(res) != 1:
            raise SegmenterError(f"expected one result for one frame, got {0 if not res else len(res)}")
        r = res[0]
        segs = []
        if r.masks is not None and len(r.masks.xy):
            conf, cls = _np(r.boxes.conf).astype(float), _np(r.boxes.cls).astype(float)
            if len(conf) != len(r.masks.xy) or len(cls) != len(conf):
                raise SegmenterError("mask / box count mismatch in segmenter output")
            for poly, c, k in zip(r.masks.xy, conf, cls):
                if not (math.isfinite(c) and 0.0 <= c <= 1.0) or k != round(k) or not 0 <= k < len(CLASS_NAMES):
                    raise SegmenterError(f"bad segment (conf {c}, class {k})")
                if len(poly):                                # Ultralytics returns [] for a sub-pixel mask
                    segs.append(polygon_segment(int(k), float(c), poly))
        segs.sort(key=lambda s: -s.confidence)
        return SegmentationResult(camera_id, frame_seq, frame_ts, tuple(segs), (w, h), ms, self.model_id, self.conf)


class FakeSeg:
    """Stand-in Ultralytics seg model for tests: fn(frame) -> list of (class_id, conf, polygon)."""

    names = {0: "bottle", 1: "cap", 2: "label"}
    task = "segment"

    def __init__(self, fn):
        self.fn = fn

    def predict(self, frame, **_):
        items = self.fn(frame)

        class _Boxes:
            conf = np.array([c for _, c, _ in items], dtype=float)
            cls = np.array([k for k, _, _ in items], dtype=float)

        class _Masks:
            xy = [np.asarray(p, dtype=np.float32) for _, _, p in items]

        class _R:
            boxes = _Boxes()
            masks = _Masks() if items else None
        return [_R()]


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def demo():
    ok, why = available()
    assert ok == DEFAULT_WEIGHTS.is_file() and (ok or "no segmentation model trained" in why), why
    try:
        YoloSegmenter()
        assert DEFAULT_WEIGHTS.is_file(), "constructed a segmenter without weights"
    except SegmenterError as e:
        assert not DEFAULT_WEIGHTS.is_file() and "no segmentation model trained" in str(e), e

    frame = np.zeros((400, 200, 3), np.uint8)
    box = {"items": [(0, 0.95, rect(50, 40, 150, 380)), (1, 0.9, rect(80, 20, 120, 45)),
                     (2, 0.85, rect(55, 150, 145, 300))]}
    seg = YoloSegmenter(model=FakeSeg(lambda f: box["items"]))
    r = seg.segment(frame, camera_id="0", frame_seq=7, frame_ts=1.0)
    m = r.measurements()
    assert [s.class_name for s in r.segments] == ["bottle", "cap", "label"]
    assert abs(m["label_area_ratio"] - (90 * 150) / (100 * 340)) < 1e-6 and m["cap_present"], m
    assert abs(m["label_fill"] - 1.0) < 1e-6
    torn = [(55, 150), (145, 150), (145, 300), (100, 220), (55, 300)]       # a notch torn out of the label
    box["items"] = [box["items"][0], (2, 0.8, torn)]
    m = seg.segment(frame).measurements()
    assert m["label_fill"] < 0.8 and m["cap_present"] is False, m
    box["items"] = [(0, 0.9, rect(50, 40, 150, 380))]
    assert seg.segment(frame).measurements()["label_area_ratio"] == 0.0         # bottle, no label: measured zero
    box["items"] = []
    m = seg.segment(frame).measurements()
    assert m["label_area_ratio"] is None and m["cap_present"] is None, m        # nothing measured: None, not 0
    for bad in ([(2, float("nan"), rect(1, 1, 5, 5))], [(5, 0.9, rect(1, 1, 5, 5))], [(0, 0.9, [(1, 1), (2, 2)])]):
        box["items"] = bad
        try:
            seg.segment(frame)
            raise AssertionError(f"bad output accepted: {bad}")
        except SegmenterError:
            pass
    try:
        seg.segment(np.zeros((4, 4), np.uint8))
        raise AssertionError("2-D frame accepted")
    except SegmenterError:
        pass
    print(f"ok  segmentation runtime interface (fake model); trained weights present: {ok}")


if __name__ == "__main__":
    demo()
