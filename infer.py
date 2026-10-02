"""Camera -> crop -> model -> verdict, plus the MJPEG frames the dashboard shows."""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np
import torch

import dataset as D

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Overlay colours, BGR, matching the dashboard theme. PASS is the theme blue.
# REJECT stays red rather than a second blue: it is the one thing an operator
# reads across a noisy room at a glance, and a colour scheme is not a reason to
# make a reject look like a pass.
C_PASS = (216, 78, 29)             # #1d4ed8
C_FAIL = (28, 28, 185)             # #b91c1c
C_TEXT = (255, 255, 255)
C_MUTED = (150, 150, 150)
C_LIGHT = (235, 235, 235)


def list_models() -> list[str]:
    if not D.MODELS.exists():
        return []
    return sorted((p.name for p in D.MODELS.iterdir() if (p / "model.pt").exists()), reverse=True)


def open_capture(src):
    """Open a webcam index or a file/URL.

    Integer sources go through DirectShow on Windows. The default MSMF backend
    takes 5-10 s to open a webcam there and sometimes just fails; DSHOW is
    near-instant. Probing and streaming must use the same backend or they
    disagree about which indices exist and at what resolution.
    """
    if isinstance(src, int) or str(src).isdigit():
        if sys.platform == "win32":
            return cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(str(src))


def list_cameras(max_index: int = 5) -> list[dict]:
    """Probe indices 0..max_index-1 for a camera that actually delivers a frame.

    isOpened() alone is not enough -- a claimed device that never returns a
    frame still reports open, and would show up as a selectable dead camera.
    """
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass                                     # noisy probe logs, not fatal
    found = []
    for i in range(max_index):
        cap = open_capture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "width": int(frame.shape[1]),
                                  "height": int(frame.shape[0])})
        finally:
            cap.release()
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
    except Exception:
        pass
    return found


class Model:
    def __init__(self, stamp: str):
        import train
        ck = torch.load(D.MODELS / stamp / "model.pt", map_location="cpu", weights_only=False)
        self.stamp, self.defects = stamp, ck["defects"]
        self.arch = ck.get("arch", train.DEFAULT_ARCH)
        # the ROI and input size are baked into the checkpoint: a model must be
        # fed the same crop it was trained on, whatever config.json says today
        self.input_wh, self.roi = tuple(ck["input_wh"]), ck.get("roi")
        self.roi_frame = ck.get("roi_frame")
        # The cache size has to come from the checkpoint too, not just the ROI.
        # model_input centre-crops a cache-sized letterbox down to input_wh, so a
        # model trained on a wide object would otherwise be fed a frame narrower
        # than its own input -- no error, just padding where the object was.
        # Checkpoints written before cache_wh existed fall back to the formula,
        # which reproduces the old (216, 496) exactly for a 192x448 input.
        self.cfg = {"roi": self.roi, "roi_frame": self.roi_frame,
                    "input_wh": list(self.input_wh), "cache_wh": ck.get("cache_wh")}
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = train.build(self.arch, len(self.defects), pretrained=False)
        m.load_state_dict(ck["state_dict"])
        self.net = m.eval().to(self.dev)

    def predict_view(self, view: np.ndarray) -> dict[str, float]:
        """Score a view that is already exactly the model's input size.

        Split out from predict() so re-scoring stored images can reach the
        network through the cached crop, which is the identical path validation
        uses. Going via predict() would re-crop an already-cropped image.
        """
        x = (view[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(self.dev)
        with torch.no_grad():
            p = torch.sigmoid(self.net(t))[0].float().cpu().numpy()
        return {d: float(v) for d, v in zip(self.defects, p)}

    def predict(self, bgr: np.ndarray) -> dict[str, float]:
        return self.predict_view(D.model_input(bgr, self.cfg, self.input_wh))


def verdict(probs: dict, thresholds: dict) -> tuple[bool, list[str]]:
    hits = [d for d, v in probs.items() if v >= float(thresholds.get(d, 0.5))]
    return (not hits), hits


class Camera:
    """One background grab thread. Inference runs on the newest frame only, so a
    slow model drops frames instead of building a latency backlog."""

    def __init__(self, name: str = "Camera"):
        self.name = name
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.probs: dict[str, float] = {}
        self.hits: list[str] = []
        self.ok = True
        self.fps = 0.0
        self.error: str | None = None
        self.model: Model | None = None
        self.source = None
        # Live performance. read_ms is how long the driver made us wait for a
        # frame, infer_ms is how long the model took on it, and dropped counts
        # frames that arrived while inference was still busy -- the three
        # numbers that say whether this pipeline keeps up with the conveyor.
        self.read_ms = 0.0
        self.infer_ms = 0.0
        self.latency_ms = 0.0
        self.dropped = 0
        self.grabbed = 0
        self.scored = 0
        self.started_at = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stats(self) -> dict:
        """The numbers the monitoring panel reads. Cheap enough to call at 4 Hz."""
        drop_pct = 100.0 * self.dropped / self.grabbed if self.grabbed else 0.0
        return {"name": self.name, "source": self.source, "fps": round(self.fps, 1),
                "read_ms": round(self.read_ms, 1), "infer_ms": round(self.infer_ms, 1),
                "latency_ms": round(self.latency_ms, 1), "dropped": self.dropped,
                "grabbed": self.grabbed, "scored": self.scored,
                "drop_pct": round(drop_pct, 1), "alive": self.alive, "error": self.error,
                "model": self.model.stamp if self.model else None,
                "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0}

    def start(self, source=0):
        if self._thread and self._thread.is_alive() and self.source == source:
            return
        self.stop()
        self.source, self._stop = source, threading.Event()
        self.dropped = self.grabbed = self.scored = 0
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def load_model(self, stamp: str | None):
        self.model = Model(stamp) if stamp else None

    def _loop(self):
        src = self.source
        cap = open_capture(src)
        if not cap.isOpened():
            self.error = f"cannot open camera source {src!r}"
            return
        self.error = None
        # A file has no natural pace: read() returns as fast as it can decode, so
        # a 30 s clip flashes past in two. Play it at its own frame rate instead.
        is_file = not str(src).isdigit()
        src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        frame_dt = 1.0 / src_fps if src_fps and src_fps > 1 else 0.0

        last, n, t0 = 0.0, 0, time.time()
        while not self._stop.is_set():
            t_frame = time.time()
            t_read = time.perf_counter()
            got, frame = cap.read()
            self.read_ms = (time.perf_counter() - t_read) * 1000
            self.grabbed += 1
            if not got:
                # a video file ran out; loop it so dry-runs repeat
                if not str(src).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "camera stopped returning frames"
                break
            now = time.time()
            probs, hits, ok = self.probs, self.hits, self.ok
            if self.model is not None and now - last > 0.06:  # ~15 Hz is plenty
                last = now
                cfg = D.load_config()
                t_inf = time.perf_counter()
                probs = self.model.predict(frame)
                self.infer_ms = (time.perf_counter() - t_inf) * 1000
                self.scored += 1
                ok, hits = verdict(probs, cfg.get("thresholds", {}))
            elif self.model is not None:
                # A frame arrived that inference did not look at. Not a fault --
                # it is how the pipeline sheds load instead of building a lag --
                # but it is the number that says whether a bottle could pass by
                # unscored, so it has to be counted rather than quietly skipped.
                self.dropped += 1
            self.latency_ms = (time.time() - t_frame) * 1000
            n += 1
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now
            with self.lock:
                self.frame, self.probs, self.hits, self.ok = frame, probs, hits, ok
            if frame_dt:
                lag = frame_dt - (time.time() - t_frame)
                if lag > 0:
                    self._stop.wait(lag)      # wait(), so Stop is still instant
        cap.release()

    def snapshot(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def overlay(self) -> bytes | None:
        """JPEG bytes, for HTTP streaming."""
        f = self.overlay_frame()
        return None if f is None else cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()

    def overlay_frame(self, width: int = 900):
        """The annotated frame as an array. A desktop UI wants this directly --
        encoding to JPEG just to decode it again is pure waste."""
        with self.lock:
            if self.frame is None:
                return None
            frame, probs, hits, ok = self.frame.copy(), dict(self.probs), list(self.hits), self.ok

        h, w = frame.shape[:2]
        s = width / max(1, w)
        frame = cv2.resize(frame, (width, max(1, round(h * s))))

        if self.model is not None and self.model.roi:
            rx, ry, rw, rh = self.model.roi
            rf = self.model.roi_frame
            if rf and rf[0] and (rf[0], rf[1]) != (w, h):   # ROI scales with the frame
                fx, fy = w / rf[0], h / rf[1]
                rx, ry, rw, rh = rx * fx, ry * fy, rw * fx, rh * fy
            x, y = round(rx * s), round(ry * s)
            cv2.rectangle(frame, (x, y), (x + round(rw * s), y + round(rh * s)), (90, 90, 90), 1)

        if probs:
            colour = C_PASS if ok else C_FAIL
            text = "PASS" if ok else f"REJECT ({len(hits)})"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), colour, -1)
            cv2.putText(frame, text, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TEXT, 2)
            cv2.putText(frame, f"{self.name}  {self.fps:.0f} fps  {self.infer_ms:.0f} ms",
                        (frame.shape[1] - 260, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
            y0 = 62
            for d, v in sorted(probs.items(), key=lambda kv: -kv[1]):
                on = d in hits
                cv2.rectangle(frame, (12, y0), (12 + int(150 * v), y0 + 13),
                              C_FAIL if on else C_MUTED, -1)
                cv2.putText(frame, f"{d} {v:.2f}", (170, y0 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            C_FAIL if on else C_LIGHT, 1)
                y0 += 19
        else:
            cv2.putText(frame, f"{self.name}: no model loaded - train one first", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_FAIL, 2)
        return frame


class CameraSet:
    """Several cameras at once, each scoring independently.

    The combined verdict is REJECT if ANY camera rejects. That is the right rule
    for the usual multi-camera layout -- two or three angles of the same bottle,
    where a defect only the side camera can see is still a defect, and requiring
    agreement would mean the extra cameras could only ever overrule a reject.
    If the cameras watch independent lines instead, read each pane's own verdict
    and ignore the combined one.
    """

    def __init__(self):
        self.cams: dict[str, Camera] = {}

    # A source may be an int index or a file path; str() keys both the same way
    # the UI labels them, so re-adding the same source replaces rather than
    # silently opens a second capture on one device.
    @staticmethod
    def key(source) -> str:
        return str(source)

    def add(self, source, name: str | None = None) -> Camera:
        k = self.key(source)
        if k not in self.cams:
            self.cams[k] = Camera(name or f"Camera {k}")
        elif name:
            self.cams[k].name = name
        return self.cams[k]

    def get(self, source) -> Camera | None:
        return self.cams.get(self.key(source))

    def running(self) -> list[Camera]:
        return [c for c in self.cams.values() if c.alive]

    def primary(self) -> Camera | None:
        return next(iter(self.running()), None)

    def load_model(self, stamp: str | None):
        """One Model object shared by every camera.

        Loading it per camera would put a second and third copy of the weights
        on the GPU for no benefit -- predict() holds no per-camera state.
        """
        model = Model(stamp) if stamp else None
        for c in self.cams.values():
            c.model = model

    def start(self, sources, stamp: str | None = None):
        """Run exactly these sources; anything else already running is stopped."""
        wanted = {self.key(s): s for s in sources}
        for k, c in list(self.cams.items()):
            if k not in wanted and c.alive:
                c.stop()
        model = Model(stamp) if stamp else None
        for k, s in wanted.items():
            cam = self.add(s)
            cam.model = model
            cam.start(s)
        return self.running()

    def stop(self):
        for c in self.cams.values():
            c.stop()

    def stats(self) -> list[dict]:
        return [c.stats() for c in self.cams.values() if c.alive]

    def combined(self) -> tuple[bool, dict[str, list[str]]]:
        """(pass, {camera name: defects it flagged}). No camera running = no claim."""
        live = [c for c in self.running() if c.probs]
        if not live:
            return True, {}
        by_cam = {c.name: list(c.hits) for c in live if c.hits}
        return (not by_cam), by_cam


def demo():
    """Self-check on the verdict rule -- the branch that decides pass/fail."""
    probs = {"a": 0.9, "b": 0.2, "c": 0.55}
    ok, hits = verdict(probs, {"a": 0.5, "b": 0.5, "c": 0.5})
    assert ok is False and hits == ["a", "c"], (ok, hits)
    ok, hits = verdict(probs, {"a": 0.95, "b": 0.5, "c": 0.6})
    assert ok is True and hits == [], (ok, hits)
    # a defect with no configured threshold must still default to 0.5, not pass
    ok, hits = verdict({"z": 0.7}, {})
    assert ok is False and hits == ["z"]
    assert verdict({}, {}) == (True, [])

    # probing must never raise, whatever hardware is or is not attached
    assert list_cameras(0) == []
    cams = list_cameras(2)
    assert isinstance(cams, list) and all(c["width"] > 0 for c in cams), cams

    # multi-camera: ANY camera rejecting rejects the bottle. Built without
    # opening a device -- the fusion rule is the part that can be wrong.
    cs = CameraSet()
    a, b = cs.add(0, "front"), cs.add("clip.mp4", "side")
    assert cs.add(0) is a, "same source opened a second capture"
    assert len(cs.cams) == 2
    assert cs.combined() == (True, {}), "nothing running must not claim a pass"
    for c in (a, b):
        c._thread = threading.current_thread()      # pretend both are live
        c.probs = {"tilt_cap": 0.1}
    assert cs.combined() == (True, {})
    b.hits = ["tilt_cap"]
    ok, by = cs.combined()
    assert ok is False and by == {"side": ["tilt_cap"]}, (ok, by)
    a.hits = ["water_level"]
    ok, by = cs.combined()
    assert ok is False and set(by) == {"front", "side"}, by
    assert cs.primary() is a
    st = a.stats()
    assert st["name"] == "front" and st["alive"] and st["drop_pct"] == 0.0, st
    a.grabbed, a.dropped = 100, 25
    assert a.stats()["drop_pct"] == 25.0
    for c in (a, b):
        c._thread = None

    print(f"ok  ({len(cams)} camera(s) found on indices 0-1; "
          f"multi-camera fusion + perf counters checked)")


if __name__ == "__main__":
    demo()
