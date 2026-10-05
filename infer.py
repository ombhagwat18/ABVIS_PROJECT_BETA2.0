"""Camera -> crop -> model -> verdict, plus the MJPEG frames the dashboard shows."""
from __future__ import annotations

import collections
import math
import sys
import threading
import time
from typing import NamedTuple

import cv2
import numpy as np
import torch

import dataset as D
import detect
import theme
import verdict as V

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Overlay colours, BGR, from theme.py: the same green / red / amber that mean
# PASS / REJECT / FAULT everywhere else in the app. REJECT is the one thing an
# operator reads across a noisy room at a glance; FAULT ("not inspected") is
# distinct from both.
C_PASS = theme.bgr(theme.PASS)
C_FAIL = theme.bgr(theme.REJECT)
C_FAULT = theme.bgr(theme.FAULT)
# component boxes (BGR): bottle, cap, label -- functional, not a design decision
C_BOX = {"bottle": (230, 160, 40), "cap": (60, 200, 60), "label": (40, 200, 230)}
C_TEXT = (255, 255, 255)
C_MUTED = (150, 150, 150)
C_LIGHT = (235, 235, 235)


def list_models() -> list[str]:
    if not D.MODELS.exists():
        return []
    return sorted((p.name for p in D.MODELS.iterdir() if (p / "model.pt").exists()), reverse=True)


def open_capture(src, size=None, fourcc=None):
    """Open a webcam index or a file/URL.

    Integer sources go through DirectShow on Windows. The default MSMF backend
    takes 5-10 s to open a webcam there and sometimes just fails; DSHOW is
    near-instant. Probing and streaming must use the same backend or they
    disagree about which indices exist and at what resolution.

    size=(w, h) / fourcc="MJPG" request a capture mode from a webcam. Without them
    DirectShow opens at the driver default (640x480 on the EMEET Nova 4K); 1920x1080
    needs MJPG to reach 30 fps over USB. A request is not a guarantee: read the
    delivered frame's shape (Camera reports it).
    """
    if isinstance(src, int) or str(src).isdigit():
        cap = (cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if sys.platform == "win32"
               else cv2.VideoCapture(int(src)))
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)[:4].ljust(4)))
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
        ctl = camera_controls(src)
        if ctl:
            applied_controls[str(int(src))] = apply_controls(cap, ctl)
        return cap
    return cv2.VideoCapture(str(src))


# Driver controls a webcam can be locked to: settings.json "camera_controls" =
# {"<index>": {"autofocus": 0, "focus": 30, "auto_exposure": 0.25, "exposure": -6, "auto_wb": 0,
# "wb_temperature": 6500, "rotate": 90}}. On a moving bottle, autofocus hunting and auto-exposure
# drift change the picture from bottle to bottle. Values are passed to the driver as-is (their scale
# is driver-specific); what the driver actually took is read back into applied_controls, because a
# webcam often ignores a property without any error. "rotate" (0/90/180/270, clockwise) is not a
# driver property: Camera turns each frame before it gets a seq number, so every consumer
# (Live, Production, dataset capture, --bench) sees the same, already-rotated frame.
CAMERA_PROPS = {"auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "exposure": cv2.CAP_PROP_EXPOSURE,
                "autofocus": cv2.CAP_PROP_AUTOFOCUS, "focus": cv2.CAP_PROP_FOCUS,
                "gain": cv2.CAP_PROP_GAIN, "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "auto_wb": cv2.CAP_PROP_AUTO_WB, "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE}
_AUTO_FIRST = ("auto_exposure", "autofocus", "auto_wb")   # a manual value is ignored while auto is on
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
applied_controls: dict = {}        # str(index) -> {control: [requested, read back]} at the last open


def camera_controls(src) -> dict:
    """This source's entry in settings.json "camera_controls" ({} if none)."""
    c = D.load_settings().get("camera_controls") or {}
    return dict(c.get(str(src)) or {}) if isinstance(c, dict) else {}


def apply_controls(cap, controls: dict) -> dict:
    """Set the driver controls; {control: [requested, read back]} (read back None = unknown name)."""
    names = sorted((k for k in controls if k != "rotate"), key=lambda k: k not in _AUTO_FIRST)
    for k in names:
        if k in CAMERA_PROPS:
            cap.set(CAMERA_PROPS[k], float(controls[k]))
    return {k: [controls[k], round(float(cap.get(CAMERA_PROPS[k])), 2) if k in CAMERA_PROPS else None]
            for k in names}


def orient(frame, rotate):
    """frame turned clockwise by rotate degrees (0/90/180/270); anything else = unchanged."""
    r = ROTATIONS.get(int(rotate or 0) % 360)
    return frame if r is None or frame is None else cv2.rotate(frame, r)


def camera_names() -> list:
    """DirectShow video-input names in device order, which is the index order cv2.CAP_DSHOW
    uses (index i -> names[i]). [] off Windows or if the COM query fails: names are a label
    for the operator, never needed to open a camera."""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        import comtypes
        import comtypes.client
        from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
        from comtypes.automation import VARIANT
        from ctypes import POINTER, c_ulong, c_void_p
        from ctypes.wintypes import DWORD, LPCOLESTR

        class IPropertyBag(IUnknown):
            _iid_ = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")
            _methods_ = [COMMETHOD([], HRESULT, "Read", (["in"], LPCOLESTR, "name"),
                                   (["in", "out"], POINTER(VARIANT), "pVar"), (["in"], c_void_p, "pErrorLog"))]

        class IMoniker(IUnknown):
            _iid_ = GUID("{0000000F-0000-0000-C000-000000000046}")
            _methods_ = [COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID))),
                         COMMETHOD([], HRESULT, "IsDirty"),
                         COMMETHOD([], HRESULT, "Load", (["in"], c_void_p)),
                         COMMETHOD([], HRESULT, "Save", (["in"], c_void_p), (["in"], ctypes.c_int)),
                         COMMETHOD([], HRESULT, "GetSizeMax", (["out"], POINTER(ctypes.c_ulonglong))),
                         COMMETHOD([], HRESULT, "BindToObject", (["in"], c_void_p), (["in"], c_void_p),
                                   (["in"], POINTER(GUID)), (["out"], POINTER(POINTER(IUnknown)))),
                         COMMETHOD([], HRESULT, "BindToStorage", (["in"], c_void_p), (["in"], c_void_p),
                                   (["in"], POINTER(GUID)), (["out"], POINTER(POINTER(IUnknown))))]

        class IEnumMoniker(IUnknown):
            _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
            _methods_ = [COMMETHOD([], HRESULT, "Next", (["in"], c_ulong, "celt"),
                                   (["out"], POINTER(POINTER(IMoniker)), "rgelt"), (["out"], POINTER(c_ulong), "n"))]

        class ICreateDevEnum(IUnknown):
            _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            _methods_ = [COMMETHOD([], HRESULT, "CreateClassEnumerator", (["in"], POINTER(GUID)),
                                   (["out"], POINTER(POINTER(IEnumMoniker))), (["in"], DWORD))]

        comtypes.CoInitialize()
        dev_enum = comtypes.client.CreateObject(GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"),
                                                interface=ICreateDevEnum)
        found = dev_enum.CreateClassEnumerator(GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}"), 0)
        names = []
        while found:                              # NULL enumerator: no video devices at all
            mon, n = found.Next(1)
            if not n:
                break
            bag = mon.BindToStorage(None, None, IPropertyBag._iid_).QueryInterface(IPropertyBag)
            v = bag.Read("FriendlyName", VARIANT(), None)
            names.append(str(getattr(v, "value", v)))
        return names
    except Exception:                             # noqa: BLE001 - a label, never a reason to fail
        return []


_PROBE_LOCK = threading.Lock()


def list_cameras(max_index: int = 5) -> list[dict]:
    """Probe indices 0..max_index-1 for a camera that actually delivers a frame.

    isOpened() alone is not enough -- a claimed device that never returns a
    frame still reports open, and would show up as a selectable dead camera.
    Serialized: the Live and Camera tabs both scan on startup, and two threads
    enumerating DirectShow devices at once corrupts the heap (0xC0000374).
    """
    with _PROBE_LOCK:
        return _list_cameras(max_index)


def _list_cameras(max_index: int) -> list[dict]:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass                                     # noisy probe logs, not fatal
    found = []
    names = camera_names()
    for i in range(max_index):
        cap = open_capture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "width": int(frame.shape[1]),
                                  "height": int(frame.shape[0]),
                                  "name": names[i] if i < len(names) else ""})
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


# Tri-state inspection result. PASS is only ever produced from a fresh, valid
# score; anything that stops us knowing whether the bottle is good is FAULT, never
# PASS. REJECT means "inspected and defective"; FAULT means "not inspected".
PASS, REJECT, FAULT = "PASS", "REJECT", "FAULT"

# A score older than this (monotonic clock) is not evidence about the bottle in
# front of the camera now. Inference runs at ~15 Hz, so this is ~15 missed cycles.
MAX_RESULT_AGE_S = 1.0

# Frames kept per camera for per-bottle frame selection: ~0.4 s at 30 fps. Frames are
# references to arrays the grab loop already allocated, so this costs no copies.
RECENT_FRAMES = 12


class Frame(NamedTuple):
    """One captured frame plus the metadata every later stage keys on.

    ts is time.monotonic() taken the instant read() returned; seq counts captured
    frames within one camera session, from 1. The image is shared, not copied:
    treat it as read-only.
    """
    camera_id: str
    ts: float
    seq: int
    image: np.ndarray


class Inspection(NamedTuple):
    """Camera.result() plus where the answer came from. frame_* describe the newest
    frame captured, result_* the frame that was actually scored (they differ whenever
    frames were shed, so frame_seq - result_seq is the unscored backlog). Fields are
    None when nothing has been captured or scored in this session."""
    camera_id: str
    session: int
    state: str
    hits: list
    reason: str | None
    frame_seq: int | None
    frame_ts: float | None
    result_seq: int | None
    result_ts: float | None
    # Trailing and defaulted so older positional construction keeps working.
    # model_id: the checkpoint that produced the score (for a FAULT, the model
    # currently loaded, or None). infer_ms: that frame's inference time on the
    # perf_counter clock -- None unless state is PASS/REJECT.
    model_id: str | None = None
    infer_ms: float | None = None
    # Component detector (detect.YoloDetector), observational: detector -> the fresh
    # detect.DetectionResult it judged (None when OFF, FAULT or stale -- stale boxes are
    # never exposed as current). detector_state: OFF | OK | NO DETECTIONS | FAULT.
    detector: object | None = None
    detector_state: str | None = None
    detector_reason: str | None = None


# Component-detector states. NO DETECTIONS is a valid result (nothing in view), not a defect.
DET_OFF, DET_OK, DET_NONE, DET_FAULT = "OFF", "OK", "NO DETECTIONS", "FAULT"


def decide(probs: dict, thresholds: dict) -> tuple[str, list[str]]:
    """verdict() with a FAULT branch: no scores, or a non-finite score, is not a PASS.

    verdict() alone passes a NaN (NaN >= threshold is False), which is exactly the
    wrong way for a broken model to fail.
    """
    if not probs:
        return FAULT, []
    if not all(math.isfinite(float(v)) for v in probs.values()):
        return FAULT, []
    _, hits = verdict(probs, thresholds)
    return (REJECT if hits else PASS), hits


class Camera:
    """One background grab thread. Inference runs on the newest frame only, so a
    slow model drops frames instead of building a latency backlog."""

    def __init__(self, name: str = "Camera"):
        self.name = name
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.probs: dict[str, float] = {}
        self.hits: list[str] = []
        self.ok = False                # legacy flag; True only after a valid PASS score
        # Freshness is judged on the monotonic clock (immune to wall-clock steps).
        # frame_ts: when the newest frame was grabbed. result_ts: when the frame the
        # current probs/hits came from was grabbed; None = nothing valid to read.
        self.state = FAULT
        self.frame_ts: float | None = None
        self.result_ts: float | None = None
        # seq numbers the frames captured since start(), from 1; session counts
        # start()s, so (session, seq) never repeats across a restart.
        self.frame_seq = 0
        self.result_seq: int | None = None
        self.result_model: str | None = None      # stamp of the model that scored it
        self.result_infer_ms: float | None = None  # ...and how long that took
        self.session = 0
        # Optional component detector (anything with detect(frame, camera_id=, frame_seq=,
        # frame_ts=) -> detect.DetectionResult). Runs beside the Stage 1 classifier; it
        # never produces PASS. det_* mirror result_*: committed together with each frame.
        self.detector = None
        self.det_result = None
        self.det_ts: float | None = None
        self.det_frame: np.ndarray | None = None   # the frame the boxes belong to
        self.det_fault: str | None = None
        self.det_faults = 0
        self.det_ms = 0.0
        self.fault: str | None = None  # why the last inference failed, if it did
        self.faults = 0
        self.armed = False             # operator wants this camera inspecting
        self.max_age = MAX_RESULT_AGE_S
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
        # Requested capture mode for a webcam: (w, h) and a FOURCC such as "MJPG". None = driver
        # default. frame_wh is what the camera actually delivers (the request is not a promise).
        self.capture_wh: tuple | None = None
        self.fourcc: str | None = None
        self.frame_wh: tuple | None = None
        self.rotate = 0                # degrees clockwise, from settings "camera_controls" at start()
        # What the operator sees: ONE stable verdict per bottle (verdict.py), not the per-frame scores. `details`
        # switches the engineer view (component boxes, score bars) back on; it is off for every operator screen.
        self.tracker = V.VerdictTracker()
        self.details = False
        self.verdict_ts = 0.0          # monotonic time the tracker was last fed
        # The last few captured frames, oldest first, so a per-bottle inspection can pick the
        # frames taken AFTER its trigger instead of whatever happens to be newest.
        self.recent: collections.deque = collections.deque(maxlen=RECENT_FRAMES)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def frames_since(self, ts: float) -> list:
        """Captured frames (Frame tuples, oldest first) stamped at or after monotonic `ts`,
        from the current session only."""
        with self.lock:
            return [f for f in self.recent if f.ts >= ts]

    @property
    def camera_id(self) -> str:
        """Stable id: the source as CameraSet keys it, else the display name."""
        return str(self.source) if self.source is not None else self.name

    def latest_frame(self) -> Frame | None:
        """Newest captured frame with its metadata, or None before the first one."""
        with self.lock:
            if self.frame is None or self.frame_ts is None:
                return None
            return Frame(self.camera_id, self.frame_ts, self.frame_seq, self.frame)

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
                "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "session": self.session, "frame_seq": self.frame_seq, "frame_wh": self.frame_wh,
                "detector": getattr(self.detector, "model_id", None),
                "det_ms": round(self.det_ms, 1), "det_faults": self.det_faults,
                "rotate": self.rotate, "controls": applied_controls.get(str(self.source))}

    def start(self, source=0):
        if self._thread and self._thread.is_alive() and self.source == source:
            return
        self.stop()
        self.source, self._stop = source, threading.Event()
        self.rotate = int(camera_controls(source).get("rotate") or 0)
        self.dropped = self.grabbed = self.scored = self.faults = self.det_faults = 0
        self.tracker.reset()
        self.verdict_ts = 0.0
        self.started_at = time.time()
        self._invalidate()
        with self.lock:                       # a new session starts with no frame at all
            self.frame, self.frame_ts, self.frame_seq = None, None, 0
            self.frame_wh = None
            self.recent.clear()
            self.session += 1
        self.armed = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.armed = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self._invalidate()

    def _invalidate(self, fault: str | None = None):
        """Drop every score so none can be read back as a valid inspection."""
        with self.lock:
            self.probs, self.hits, self.ok = {}, [], False
            self.state, self.result_ts, self.fault = FAULT, None, fault
            self.result_seq = self.result_model = self.result_infer_ms = None
            self.det_result = self.det_ts = self.det_frame = self.det_fault = None

    def result(self, now: float | None = None) -> tuple[str, list[str], str | None]:
        """(state, defects, fault reason). PASS/REJECT only from a fresh valid score.

        Everything else -- not armed, thread dead, driver error, no model, failed
        inference, nothing scored yet, frame or score too old -- is FAULT with a
        reason. `now` is injectable so freshness can be tested without sleeping.
        """
        i = self.inspection(now)
        return i.state, i.hits, i.reason

    def inspection(self, now: float | None = None) -> Inspection:
        """result() plus the session/sequence/timestamps it was judged on, and the
        component detector's view.

        The detector is observational: it never produces PASS/REJECT. But if one is
        attached and cannot vouch for the current frame (failed, nothing detected yet,
        stale) the whole inspection is FAULT -- a camera we cannot fully see is not PASS.
        """
        now = time.monotonic() if now is None else now
        with self.lock:
            state, hits, fault = self.state, list(self.hits), self.fault
            fts, rts = self.frame_ts, self.result_ts
            fseq, rseq, session = (self.frame_seq if fts is not None else None,
                                   self.result_seq, self.session)
            rmodel, rms = self.result_model, self.result_infer_ms
            dres, dts, dfault = self.det_result, self.det_ts, self.det_fault
        detector = self.detector

        if detector is None:
            dstate, dwhy, dview = DET_OFF, None, None
        elif dfault:
            dstate, dwhy, dview = DET_FAULT, dfault, None
        elif dres is None or dts is None:
            dstate, dwhy, dview = DET_FAULT, "no detection yet", None
        elif fts is None or now - fts > self.max_age or now - dts > self.max_age:
            dstate, dwhy, dview = DET_FAULT, f"stale detections (>{self.max_age:g}s old)", None
        else:
            dstate, dwhy, dview = (DET_OK if dres.detections else DET_NONE), None, dres

        def out(st, h, why):
            if st == FAULT:                 # no valid score: nothing to time or attribute
                loaded = getattr(self.model, "stamp", None)
                return Inspection(self.camera_id, session, st, h, why, fseq, fts, rseq, rts,
                                  loaded, None, dview, dstate, dwhy)
            return Inspection(self.camera_id, session, st, h, why, fseq, fts, rseq, rts, rmodel, rms,
                              dview, dstate, dwhy)

        def base():
            if not self.armed:
                return FAULT, [], "camera not started"
            if self.error:
                return FAULT, [], self.error
            if not self.alive:
                return FAULT, [], "camera thread not running"
            if self.model is None:
                if detector is None:
                    return FAULT, [], "no model loaded"
                return FAULT, [], "detector-only test mode: no inspection rules"
            if fault:
                return FAULT, [], fault
            if rts is None or state not in (PASS, REJECT):
                return FAULT, [], "no frame scored yet"
            if fts is None or now - fts > self.max_age:
                return FAULT, [], f"stale frame (>{self.max_age:g}s old)"
            if now - rts > self.max_age:
                return FAULT, [], f"stale score (>{self.max_age:g}s old)"
            return state, hits, None

        st, h, why = base()
        if st != FAULT and dstate == DET_FAULT:
            return out(FAULT, [], f"detector: {dwhy}")
        return out(st, h, why)

    def load_model(self, stamp: str | None):
        self.model = Model(stamp) if stamp else None

    def _loop(self):
        """Thread entry. Whatever ends the loop -- a driver error, a bug, a clean stop --
        the scores are dropped on the way out, so a dead camera can never leave its
        last PASS/REJECT behind as if it were current."""
        # Pinned to this thread's own stop event: stop() gives up joining after 2 s,
        # so a wedged driver can wake up after a restart. start() swaps in a fresh
        # event, which is how that old thread knows it no longer owns the camera.
        stop = self._stop
        try:
            self._run(stop)
        except Exception as e:                       # noqa: BLE001 - last line of defence
            if stop is self._stop:
                self.error = f"camera loop crashed: {type(e).__name__}: {e}"
        finally:
            if stop is self._stop:
                self._invalidate()

    def _infer_guard(self, stop):
        """Thread entry for _infer_loop. If scoring dies for any reason the scores are dropped (the camera goes
        FAULT on 'stale score'), never left standing as a last good PASS."""
        try:
            self._infer_loop(stop)
        except Exception as e:                           # noqa: BLE001 - last line of defence
            if stop is self._stop:
                self._invalidate(f"scoring thread crashed: {type(e).__name__}: {e}")

    def _run(self, stop):
        src = self.source
        cap = open_capture(src, self.capture_wh, self.fourcc)
        if not cap.isOpened():
            self.error = f"cannot open camera source {src!r}"
            return
        self.error = None
        threading.Thread(target=self._infer_guard, args=(stop,), daemon=True).start()
        try:
            self._grab_loop(cap, src, stop)
        finally:
            cap.release()

    def _grab_loop(self, cap, src, stop):
        # A file has no natural pace: read() returns as fast as it can decode, so
        # a 30 s clip flashes past in two. Play it at its own frame rate instead.
        is_file = not str(src).isdigit()
        src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        frame_dt = 1.0 / src_fps if src_fps and src_fps > 1 else 0.0

        n, t0 = 0, time.time()
        while not stop.is_set():
            t_frame = time.time()
            t_read = time.perf_counter()
            got, frame = cap.read()
            t_grab = time.monotonic()          # the capture instant: stamped before any bookkeeping
            self.read_ms = (time.perf_counter() - t_read) * 1000
            self.grabbed += 1
            if not got:
                # a video file ran out; loop it so dry-runs repeat
                if not str(src).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "camera stopped returning frames"
                break
            if stop.is_set() or stop is not self._stop:
                break                          # superseded while read() was blocked
            if self.rotate:
                frame = orient(frame, self.rotate)
            seq = self.frame_seq + 1           # only the owning thread writes frame_seq
            now = time.time()
            n += 1
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now
            # Only the frame is committed here. Scoring runs on its own thread (_infer_loop): when it ran inline
            # the camera could not take the next frame until the model had finished with the last one, so the
            # driver's buffer filled with old frames and the picture lagged the real bottle by seconds.
            with self.lock:
                self.frame, self.frame_ts, self.frame_seq = frame, t_grab, seq
                self.frame_wh = (int(frame.shape[1]), int(frame.shape[0]))
                self.recent.append(Frame(self.camera_id, t_grab, seq, frame))
            if frame_dt:
                lag = frame_dt - (time.time() - t_frame)
                if lag > 0:
                    stop.wait(lag)            # wait(), so Stop is still instant

    def _infer_loop(self, stop):
        """Score the NEWEST captured frame, at most ~15 Hz, on a thread of its own.

        A result carries the seq and capture time of the frame it scored, exactly as before, so freshness
        (result_ts) and the FAULT rules are unchanged. Frames that arrive while a score is running are skipped
        (counted in `dropped`), never queued: the model always looks at the newest picture, not a backlog."""
        last_seq, last_t = 0, 0.0
        while not stop.is_set() and stop is self._stop:
            with self.lock:
                frame, fts, fseq = self.frame, self.frame_ts, self.frame_seq
            model, detector = self.model, self.detector
            if frame is None or fseq == last_seq:
                stop.wait(0.004)
                continue
            gap = time.time() - last_t
            if gap < 0.06:                               # ~15 Hz is plenty
                stop.wait(0.06 - gap)
                continue
            if last_seq:
                self.dropped += max(0, fseq - last_seq - 1)
            last_seq, last_t = fseq, time.time()
            with self.lock:
                probs, hits, ok = self.probs, self.hits, self.ok
                state, result_ts, fault = self.state, self.result_ts, self.fault
                result_seq = self.result_seq
                result_model, result_ms = self.result_model, self.result_infer_ms
                det_result, det_ts, det_frame, det_fault = (self.det_result, self.det_ts,
                                                            self.det_frame, self.det_fault)
            if model is None:
                probs, hits, ok, state, result_ts = {}, [], False, FAULT, None
                result_seq = result_model = result_ms = None
            if detector is None:
                det_result = det_ts = det_frame = det_fault = None
            if model is not None:
                t_inf = time.perf_counter()
                try:
                    cfg = D.load_config()
                    probs = model.predict(frame)
                    self.infer_ms = (time.perf_counter() - t_inf) * 1000
                    state, hits = decide(probs, cfg.get("thresholds", {}))
                    if state == FAULT:
                        raise ValueError("model returned no usable scores")
                    ok, result_ts, fault = state == PASS, fts, None
                    result_seq, result_ms = fseq, self.infer_ms
                    result_model = getattr(model, "stamp", None)
                    self.scored += 1
                except Exception as e:                    # noqa: BLE001 - must not kill the loop
                    # Keep grabbing, but drop the scores: this frame was not inspected.
                    probs, hits, ok, state, result_ts = {}, [], False, FAULT, None
                    result_seq = result_model = result_ms = None
                    fault = f"inference failed: {type(e).__name__}: {e}"
                    self.faults += 1
            if detector is not None:
                t_det = time.perf_counter()
                try:
                    det_result = detector.detect(frame, camera_id=self.camera_id,
                                                 frame_seq=fseq, frame_ts=fts)
                    self.det_ms = (time.perf_counter() - t_det) * 1000
                    det_ts, det_frame, det_fault = fts, frame, None
                except Exception as e:                    # noqa: BLE001 - must not kill the loop
                    det_result = det_ts = det_frame = None
                    det_fault = f"detection failed: {type(e).__name__}: {e}"
                    self.det_faults += 1
            if stop.is_set() or stop is not self._stop:
                return                                    # the camera was stopped or restarted while scoring
            with self.lock:
                self.result_seq = result_seq
                self.result_model, self.result_infer_ms = result_model, result_ms
                self.probs, self.hits, self.ok = probs, hits, ok
                self.state, self.result_ts, self.fault = state, result_ts, fault
                self.det_result, self.det_ts = det_result, det_ts
                self.det_frame, self.det_fault = det_frame, det_fault
            self._feed_tracker(state, hits, fault, model, detector, det_result, det_fault)
            self.latency_ms = (time.monotonic() - fts) * 1000    # capture -> result

    def _feed_tracker(self, state, hits, fault, model, detector, det_result, det_fault):
        """One call per scored frame: what this frame says about THIS bottle, for the stable verdict."""
        defects = list(hits)
        present = None
        valid = model is not None and state in (PASS, REJECT) and not fault
        if model is None and detector is not None:
            valid = True                                      # detector-only view: no classifier defects
        why = fault or ("no model loaded" if model is None and detector is None else "")
        if detector is not None:
            if det_fault or det_result is None:
                valid, why = False, det_fault or "no detection yet"
            else:
                import decision as DEC                        # lazy: decision imports this module
                cfg = D.load_config()
                rules = dict(DEC.RULES)
                if cfg.get("inspection"):
                    rules["recipe"] = cfg["inspection"]
                found, _ = DEC.detection_findings(det_result, rules)
                present = found is not None                   # a bottle (the recipe's anchor) is in view
                if found:
                    defects += [d for d in found if d not in defects]
        if present is False:
            defects = []                                      # nothing to judge: scores on an empty belt are noise
        self.tracker.update(valid, defects, present, why)
        self.verdict_ts = time.monotonic()

    def display_verdict(self) -> "V.Verdict":
        """The stable verdict, or FAULT when this camera is not producing results (stopped, dead, stale)."""
        if not self.armed:
            return V.Verdict(V.FAULT, reason="camera off")
        if self.error:
            return V.Verdict(V.FAULT, reason=self.error)
        if not self.alive:
            return V.Verdict(V.FAULT, reason="camera not running")
        if self.verdict_ts and time.monotonic() - self.verdict_ts > max(2.0, self.max_age * 2):
            return V.Verdict(V.FAULT, reason="no new result")
        return self.tracker.current()

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
            frame, probs, hits = self.frame.copy(), dict(self.probs), list(self.hits)
            dframe = self.det_frame
        insp = self.inspection()               # PASS/REJECT only if the score is fresh
        state, reason, dres = insp.state, insp.reason, insp.detector
        if (dres is not None and dframe is not None
                and dframe.shape[:2] == (dres.frame_wh[1], dres.frame_wh[0])):
            frame = dframe.copy()              # boxes are drawn on the frame they belong to

        h, w = frame.shape[:2]
        s = width / max(1, w)
        frame = cv2.resize(frame, (width, max(1, round(h * s))))
        v = self.display_verdict()
        if not self.details:
            return self._industrial_overlay(frame, v)

        # ---- engineer view: component boxes, score bars, detector line (Live tab "Engineer details")
        if self.model is not None and self.model.roi:
            rx, ry, rw, rh = self.model.roi
            rf = self.model.roi_frame
            if rf and rf[0] and (rf[0], rf[1]) != (w, h):   # ROI scales with the frame
                fx, fy = w / rf[0], h / rf[1]
                rx, ry, rw, rh = rx * fx, ry * fy, rw * fx, rh * fy
            x, y = round(rx * s), round(ry * s)
            cv2.rectangle(frame, (x, y), (x + round(rw * s), y + round(rh * s)), (90, 90, 90), 1)

        if dres is not None:                   # component boxes: coordinates are original-frame pixels
            fh2, fw2 = frame.shape[:2]
            for d in dres.detections:
                x1, y1, x2, y2 = (round(q) for q in
                                  detect.scale_box((d.x1, d.y1, d.x2, d.y2), dres.frame_wh, (fw2, fh2)))
                col = C_BOX.get(d.class_name, C_MUTED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}",
                            (x1 + 4, max(y1 + 18 + 18 * d.class_id, 66)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        self._banner(frame, v, 54)
        if probs:
            cv2.putText(frame, f"{self.name}  {self.fps:.0f} fps  {self.infer_ms:.0f} ms",
                        (frame.shape[1] - 260, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
            y0 = 68
            for d, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                on = d in hits
                cv2.rectangle(frame, (12, y0), (12 + int(150 * p), y0 + 13), C_FAIL if on else C_MUTED, -1)
                cv2.putText(frame, f"{d} {p:.2f}", (170, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            C_FAIL if on else C_LIGHT, 1)
                y0 += 19
        if insp.detector_state != DET_OFF:
            dtxt = {DET_OK: "DETECTOR OK  " + "  ".join(f"{n} {c}" for n, c in dres.counts().items())
                    + f"  {dres.infer_ms:.0f} ms  (dev conf {dres.conf_threshold:g})" if dres else "",
                    DET_NONE: "DETECTOR: no components detected (not a defect)"
                    }.get(insp.detector_state, f"DETECTOR FAULT: {insp.detector_reason}")
            bh = frame.shape[0]
            cv2.rectangle(frame, (0, bh - 26), (frame.shape[1], bh), (30, 30, 30), -1)
            cv2.putText(frame, dtxt, (10, bh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        C_FAULT if insp.detector_state == DET_FAULT else C_TEXT, 1)
        return frame

    # verdict colours (BGR): GOOD green, DEFECT red, FAULT amber, the rest neutral
    _VCOL = {V.GOOD: C_PASS, V.DEFECT: C_FAIL, V.FAULT: C_FAULT, V.CHECKING: (170, 120, 60), V.EMPTY: (90, 90, 90)}

    def _banner(self, frame, v, height):
        col = self._VCOL.get(v.kind, C_FAULT)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], height), col, -1)
        text = v.text
        scale = height / 46.0
        while scale > 0.5 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > frame.shape[1] - 24:
            scale -= 0.1
        cv2.putText(frame, text, (14, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, C_TEXT, 2)

    def _industrial_overlay(self, frame, v):
        """What an operator needs and nothing else: one verdict in words, a frame in its colour, the camera name.
        No component boxes, no score bars, no detector line, no per-frame flicker (verdict.py latches)."""
        h, w = frame.shape[:2]
        col = self._VCOL.get(v.kind, C_FAULT)
        if v.kind in (V.GOOD, V.DEFECT, V.FAULT):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 6)
        self._banner(frame, v, max(46, h // 9))
        cv2.putText(frame, self.name, (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
        return frame


class CameraSet:
    """Several cameras at once, each scoring independently.

    The combined verdict is tri-state (see combined()): REJECT if ANY camera rejects,
    FAULT if any armed camera cannot vouch for its bottle. That is the right rule
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

    def load_detector(self, detector):
        """One detector (detect.YoloDetector) shared by every camera, or None to remove it.
        Its calls are serialised internally, so sharing is safe. Cameras still run
        independently: nothing here synchronises them."""
        for c in self.cams.values():
            c.detector = detector

    def start(self, sources, stamp: str | None = None, detector=None):
        """Run exactly these sources; anything else already running is stopped.

        stamp selects the Stage 1 classifier checkpoint (None = no classifier);
        detector is an optional component detector (None = none). With only a detector
        the cameras run in detector-only test mode and report FAULT, by design: boxes
        are not an inspection verdict."""
        wanted = {self.key(s): s for s in sources}
        for k, c in list(self.cams.items()):
            if k not in wanted and c.alive:
                c.stop()
        model = Model(stamp) if stamp else None
        for k, s in wanted.items():
            cam = self.add(s)
            cam.model = model
            cam.detector = detector
            cam.start(s)
        return self.running()

    def stop(self):
        for c in self.cams.values():
            c.stop()

    def stats(self) -> list[dict]:
        return [c.stats() for c in self.cams.values() if c.alive]

    def set_details(self, on: bool):
        """Engineer view (boxes, score bars) on every camera; off = the operator's one-verdict view."""
        for c in self.cams.values():
            c.details = bool(on)

    def display_verdict(self) -> "V.Verdict":
        """ONE verdict for the bottle in front of all armed cameras (verdict.combine)."""
        armed = [c for c in self.cams.values() if c.armed]
        return V.combine([c.display_verdict() for c in armed]) if armed else V.Verdict(V.FAULT, reason="no camera running")

    def combined(self, now: float | None = None) -> tuple[str, dict[str, list[str]]]:
        """(PASS | REJECT | FAULT, detail). PASS only if every armed camera has a fresh PASS.

        detail maps camera name -> its defects (REJECT) or ["FAULT: <reason>"].
        FAULT outranks REJECT: a camera we cannot trust means the verdict is
        degraded even when another camera has already seen a defect, and the
        operator should see that. Either way nothing but PASS is a pass. No armed
        camera at all is a FAULT -- no inspection was performed.
        """
        armed = [c for c in self.cams.values() if c.armed]
        if not armed:
            return FAULT, {"(none)": ["FAULT: no camera running"]}
        detail, states = {}, set()
        for c in armed:
            state, hits, reason = c.result(now)
            states.add(state)
            if state == FAULT:
                detail[c.name] = [f"FAULT: {reason}"]
            elif hits:
                detail[c.name] = hits
        return (FAULT if FAULT in states else REJECT if REJECT in states else PASS), detail


def _selftest_tri_state():
    """PASS / REJECT / FAULT, against a real Camera thread fed a fake capture and a
    fake model -- no device, no weights. The point is that every way of not knowing
    ends in FAULT, and that a dead or wedged camera cannot leave a PASS behind."""
    me = sys.modules[__name__]
    box = {"cap": "ok", "infer": "pass"}
    release = threading.Event()

    class Cap:
        def __init__(self, mode):
            self.mode, self.n = mode, 0
        def isOpened(self):
            return self.mode != "noopen"
        def get(self, *_):
            return 0
        def set(self, *_):
            pass
        def release(self):
            pass
        def read(self):
            time.sleep(0.004)
            self.n += 1
            if self.mode == "dead":
                return False, None
            if self.mode == "raise" and self.n > 3:
                raise RuntimeError("driver exploded")
            if self.mode == "stall" and self.n > 3:
                release.wait(10)                    # wedged driver: read() never returns
            img = np.zeros((8, 8, 3), np.uint8)
            img[0, 0, 0] = self.n % 256        # tag: which read() produced this frame
            return True, img

    class FakeModel:
        defects, roi, stamp = ["a"], None, "fake"
        def predict(self, frame):
            kind = box["infer"]
            if kind == "raise":
                raise RuntimeError("cuda out of memory")
            return {"a": {"pass": 0.1, "reject": 0.9, "nan": float("nan")}[kind]}

    def wait(cond, what, timeout=3.0):
        t0 = time.monotonic()
        while not cond():
            assert time.monotonic() - t0 < timeout, f"timed out waiting for: {what}"
            time.sleep(0.01)

    def fresh(mode):
        box["cap"], box["infer"] = mode, "pass"
        cam = Camera("t")
        cam.model = FakeModel()
        cs = CameraSet()
        cs.cams["0"] = cam
        cam.start(0)
        return cam, cs

    real_open, real_cfg = me.open_capture, D.load_config
    me.open_capture = lambda src, *_: Cap(box["cap"])
    D.load_config = lambda: {"thresholds": {}}
    try:
        # -- nothing inspected -> FAULT, never PASS
        assert CameraSet().combined()[0] == FAULT, "no cameras must be FAULT"
        idle = Camera("idle"); idle.model = FakeModel()
        cs0 = CameraSet(); cs0.cams["0"] = idle
        assert cs0.combined()[0] == FAULT, "a camera that was never started must be FAULT"
        idle.armed, idle._thread = True, threading.current_thread()   # live, but nothing scored
        st, _, why = idle.result()
        assert st == FAULT and why == "no frame scored yet", (st, why)
        assert cs0.combined()[0] == FAULT
        idle._thread = None
        assert decide({}, {})[0] == FAULT and decide({"a": float("nan")}, {})[0] == FAULT
        assert decide({"a": 0.1}, {})[0] == PASS and decide({"a": 0.9}, {}) == (REJECT, ["a"])

        # -- valid frames: classification is unchanged
        cam, cs = fresh("ok")
        wait(lambda: cam.result()[0] == PASS, "valid PASS")
        assert cam.result() == (PASS, [], None) and cs.combined() == (PASS, {})
        box["infer"] = "reject"
        wait(lambda: cam.result()[0] == REJECT, "valid REJECT")
        assert cam.result()[1] == ["a"] and cs.combined() == (REJECT, {"t": ["a"]})

        # -- stale score -> FAULT (clock injected: no sleeping)
        st, _, why = cam.result(now=time.monotonic() + 10)
        assert st == FAULT and "stale" in why, (st, why)
        assert cs.combined(now=time.monotonic() + 10)[0] == FAULT

        # -- inference exception -> FAULT, thread survives, scores dropped, then recovers
        box["infer"] = "raise"
        wait(lambda: cam.result()[0] == FAULT, "inference exception")
        assert cam.alive and cam.probs == {} and cam.hits == [] and cam.faults >= 1
        assert "inference failed" in cam.result()[2] and cs.combined()[0] == FAULT
        box["infer"] = "nan"                         # a broken model must not read as PASS
        time.sleep(0.2)
        assert cam.result()[0] == FAULT
        box["infer"] = "pass"
        wait(lambda: cam.result()[0] == PASS, "recovery after a good frame")

        # -- missing model -> FAULT
        good, cam.model = cam.model, None
        assert cam.result()[0] == FAULT and cam.result()[2] == "no model loaded"
        cam.model = good
        wait(lambda: cam.result()[0] == PASS, "model restored")
        cam.stop()
        assert cam.result()[0] == FAULT and cam.probs == {}, "stopped camera kept a score"
        assert cs.combined()[0] == FAULT

        # -- camera failure -> FAULT, and the last result does not outlive the thread
        for mode, err in (("dead", "stopped returning"), ("noopen", "cannot open"),
                          ("raise", "loop crashed")):
            cam, cs = fresh(mode)
            wait(lambda: not cam.alive, f"{mode}: thread to end")
            st, _, why = cam.result()
            assert st == FAULT and err in why, (mode, st, why)
            assert cam.probs == {} and cam.hits == [] and cs.combined()[0] == FAULT
            cam.stop()

        # -- wedged driver: thread alive but frames stopped -> goes FAULT by age alone
        cam, cs = fresh("stall")
        cam.max_age = 0.4
        wait(lambda: cam.result()[0] == PASS, "pass before stall")
        wait(lambda: cam.result()[0] == FAULT, "stale after stall")
        assert cam.alive and "stale" in cam.result()[2]
        release.set(); cam.stop()

        # -- frame metadata: camera_id / monotonic ts / sequence number
        cam, cs = fresh("ok")
        t_lo = time.monotonic()
        wait(lambda: cam.latest_frame() is not None, "first frame")
        f0 = cam.latest_frame()
        assert isinstance(f0, Frame) and f0.camera_id == "0" and isinstance(f0.ts, float)
        assert t_lo <= f0.ts <= time.monotonic(), "ts must be on the monotonic clock"
        assert f0.seq >= 1 and int(f0.image[0, 0, 0]) == f0.seq % 256,             "seq must count captured frames 1:1, starting at 1"
        seen = [f0]
        for _ in range(30):
            wait(lambda: cam.latest_frame().seq > seen[-1].seq, "next frame")
            f = cam.latest_frame()
            # time.monotonic() ticks every ~15.6 ms on Windows, so neighbouring frames
            # can share a stamp; it must never go backwards. seq is the strict order.
            assert f.ts >= seen[-1].ts, "timestamps must never go backwards"
            assert int(f.image[0, 0, 0]) == f.seq % 256
            seen.append(f)
        assert [f.seq for f in seen] == sorted(f.seq for f in seen)
        assert seen[-1].ts > seen[0].ts, "timestamps never advanced across 30 frames"
        wait(lambda: cam.result()[0] == PASS, "pass with metadata")
        i = cam.inspection()
        assert (i.state, i.reason, i.camera_id, i.session) == (PASS, None, "0", cam.session), i
        assert 1 <= i.result_seq <= i.frame_seq and i.result_ts <= i.frame_ts, i
        assert cam.result() == (PASS, [], None), "result() must keep its 3-tuple shape"
        assert i.model_id == "fake" and i.infer_ms is not None and i.infer_ms >= 0, i
        box["infer"] = "reject"
        wait(lambda: cam.result()[0] == REJECT, "reject with metadata")
        assert cam.inspection().hits == ["a"]
        stale = cam.inspection(now=time.monotonic() + 10)       # stale still FAULTs, keeps its ids
        assert stale.state == FAULT and "stale" in stale.reason and stale.frame_seq >= 1, stale
        assert stale.infer_ms is None, "a FAULT must not carry a timing for a score it disowns"
        box["infer"] = "pass"

        # -- restart: new session, sequence starts over, nothing carried across
        wait(lambda: cam.frame_seq >= 20, "20 frames before restart")
        old_session, old_seq = cam.session, cam.frame_seq
        cam.stop(); cam.start(0)
        assert cam.session == old_session + 1
        assert cam.frame_seq < old_seq and cam.frame_seq <= 10, (old_seq, cam.frame_seq)
        j = cam.inspection()
        assert j.state == FAULT and j.result_seq is None, "restart must not inherit a score"
        wait(lambda: cam.latest_frame() is not None, "first frame of new session")
        g = cam.latest_frame()
        assert int(g.image[0, 0, 0]) == g.seq % 256 and g.seq < old_seq
        wait(lambda: cam.result()[0] == PASS, "pass after restart")
        cam.stop()

        # -- a wedged old thread that wakes after a restart must not touch the new session
        release.clear()                           # the earlier stall test left it set
        cam, cs = fresh("stall")
        wait(lambda: cam.frame_seq >= 3, "stall reached")
        old = cam._thread
        box["cap"] = "ok"
        cam.stop(); cam.start(0)                  # old thread is stuck in read(): stop() times out
        sess = cam.session
        release.set()
        wait(lambda: not old.is_alive(), "old thread to exit")
        time.sleep(0.15)
        last = cam.latest_frame().seq
        wait(lambda: cam.latest_frame().seq > last, "new session keeps counting")
        f = cam.latest_frame()
        assert cam.session == sess and int(f.image[0, 0, 0]) == f.seq % 256,             "old thread corrupted the new session's frames"
        wait(lambda: cam.result()[0] == PASS, "new session still inspects")
        cam.stop()
        release.clear()

        # -- fusion across cameras: FAULT > REJECT > PASS; only all-PASS passes
        def armed_cam(name, state, hits=()):
            c = Camera(name); c.model, c.armed = FakeModel(), True
            c._thread = threading.current_thread()
            c.state, c.hits = state, list(hits)
            c.frame_ts = c.result_ts = time.monotonic()
            return c
        cs = CameraSet()
        a, b = armed_cam("front", PASS), armed_cam("side", PASS)
        cs.cams.update({"0": a, "1": b})
        assert cs.combined() == (PASS, {})
        b.state, b.hits = REJECT, ["tilt_cap"]
        assert cs.combined() == (REJECT, {"side": ["tilt_cap"]})
        a.result_ts = time.monotonic() - 60           # front goes stale
        st, by = cs.combined()
        assert st == FAULT and "FAULT" in by["front"][0] and by["side"] == ["tilt_cap"], (st, by)
        for c in (a, b):
            c._thread = None
    finally:
        release.set()
        me.open_capture, D.load_config = real_open, real_cfg


def _selftest_detector():
    """Component detector inside the Camera: observational, fail-safe, and unable to turn a
    missing component into a verdict. Fake capture + fake YOLO -- no device, no weights."""
    me = sys.modules[__name__]
    box = {"cap": "ok", "yolo": "full", "model": "pass"}

    class Cap:
        def __init__(self, mode):
            self.mode, self.n = mode, 0
        def isOpened(self):
            return True
        def get(self, *_):
            return 0
        def set(self, *_):
            pass
        def release(self):
            pass
        def read(self):
            time.sleep(0.004)
            self.n += 1
            if self.mode == "dead" and self.n > 3:
                return False, None
            return True, np.zeros((36, 64, 3), np.uint8)

    class Clf:
        defects, roi, stamp = ["a"], None, "clf-1"
        def predict(self, frame):
            return {"a": 0.9 if box["model"] == "reject" else 0.1}

    def yolo_out(frame):
        k = box["yolo"]
        if k == "boom":
            raise RuntimeError("cuda out of memory")
        if k == "nan":
            return [[1, 1, 9, 9]], [float("nan")], [0]
        if k == "empty":
            return np.zeros((0, 4)), [], []
        return [[2, 3, 30, 33], [10, 4, 20, 9]], [0.93, 0.81], [0, 1]       # bottle + cap, no label

    def new_detector():
        return detect.YoloDetector(model=detect.FakeYolo(yolo_out), warmup=False)

    def wait(cond, what, timeout=3.0):
        t0 = time.monotonic()
        while not cond():
            assert time.monotonic() - t0 < timeout, f"timed out waiting for: {what}"
            time.sleep(0.01)

    def cam_with(name, source, clf=True, det=True):
        c = Camera(name)
        c.model = Clf() if clf else None
        c.detector = new_detector() if det else None
        c.start(source)
        return c

    real_open, real_cfg = me.open_capture, D.load_config
    me.open_capture = lambda src, *_: Cap(box["cap"])
    D.load_config = lambda: {"thresholds": {}}
    try:
        # -- detector alone: boxes yes, verdict never PASS
        box.update(cap="ok", yolo="full", model="pass")
        c = cam_with("solo", 0, clf=False)
        wait(lambda: c.inspection().detector_state == DET_OK, "detector OK")
        i = c.inspection()
        assert i.state == FAULT and "detector-only" in i.reason, (i.state, i.reason)
        r = i.detector
        assert r.counts() == {"bottle": 1, "cap": 1, "label": 0}, r.counts()
        assert (r.camera_id, r.frame_wh) == ("0", (64, 36)) and 1 <= r.frame_seq <= i.frame_seq, (r, i)
        assert r.detections[0].class_name == "bottle" and (r.detections[0].x1, r.detections[0].y2) == (2, 33)
        c.stop()
        assert c.inspection().detector is None, "a stopped camera kept its boxes"

        # -- classifier + detector: the Stage 1 verdict is untouched by the detector
        c = cam_with("both", 0)
        wait(lambda: c.result()[0] == PASS and c.inspection().detector_state == DET_OK, "PASS + detector")
        assert c.inspection().detector.counts()["cap"] == 1
        box["model"] = "reject"
        wait(lambda: c.result()[0] == REJECT, "classifier REJECT still works")
        assert c.result()[1] == ["a"]
        box["model"] = "pass"

        # -- absence is NOT a defect: no components detected leaves PASS as PASS
        box["yolo"] = "empty"
        wait(lambda: c.inspection().detector_state == DET_NONE, "empty detections")
        wait(lambda: c.result()[0] == PASS, "PASS survives an empty detection")
        i = c.inspection()
        assert i.state == PASS and i.detector.detections == () and i.detector_reason is None

        # -- detector failures are FAULT, never PASS; the thread survives and recovers
        for kind in ("boom", "nan"):
            box["yolo"] = kind
            wait(lambda: c.result()[0] == FAULT, f"{kind} -> FAULT")
            i = c.inspection()
            assert c.alive and i.detector is None and i.detector_state == DET_FAULT, (kind, i)
            assert i.reason.startswith("detector: detection failed"), i.reason
            assert c.det_faults >= 1
        box["yolo"] = "full"
        wait(lambda: c.result()[0] == PASS and c.inspection().detector_state == DET_OK, "recovery")

        # -- stale detections -> FAULT (clock injected)
        future = time.monotonic() + 10
        i = c.inspection(now=future)
        assert i.state == FAULT and i.detector is None and "stale" in i.reason, i
        c.stop()
        # ...and the detection age on its own: a FRESH frame with OLD detections (hand-built,
        # thread stopped so nothing refreshes them) must not read as current boxes
        c.armed, c._thread, c.model = True, threading.current_thread(), Clf()
        c.frame_ts = c.result_ts = time.monotonic()
        c.state, c.result_seq, c.frame_seq = PASS, 1, 1          # a fresh, valid Stage 1 PASS
        c.det_ts = c.frame_ts - 5.0
        c.det_result = detect.DetectionResult("0", 1, c.det_ts, (), (64, 36), 1.0, "fake", 0.25)
        i = c.inspection(now=c.frame_ts + 0.1)
        assert i.detector is None and i.detector_state == DET_FAULT and "stale detections" in i.detector_reason, i
        assert i.state == FAULT and i.reason.startswith("detector: stale"), i
        c._thread = None

        # -- camera failure drops the boxes too
        box["cap"] = "dead"
        c = cam_with("dying", 0)
        wait(lambda: not c.alive, "camera death")
        i = c.inspection()
        assert i.state == FAULT and i.detector is None and c.det_result is None, i
        c.stop()

        # -- restart: no boxes carried across sessions
        box["cap"] = "ok"
        c = cam_with("again", 0)
        wait(lambda: c.inspection().detector_state == DET_OK, "first session")
        s1 = c.session
        c.stop(); c.start(0)
        assert c.session == s1 + 1 and c.det_result is None, "restart kept old detections"
        wait(lambda: c.inspection().detector_state == DET_OK, "second session")
        assert c.inspection().detector.frame_seq <= c.frame_seq
        c.stop()

        # -- several cameras, one shared detector: independent, ids preserved, not synchronised
        shared = new_detector()
        cs = CameraSet()
        cs.start([0, 1], stamp=None, detector=shared)
        a, b = cs.get(0), cs.get(1)
        for cam in (a, b):
            cam.model = Clf()
        wait(lambda: all(cam.result()[0] == PASS and cam.inspection().detector_state == DET_OK
                         for cam in (a, b)), "two cameras OK")
        ia, ib = a.inspection(), b.inspection()
        assert (ia.detector.camera_id, ib.detector.camera_id) == ("0", "1")
        assert (ia.session, ib.session) == (a.session, b.session)
        assert cs.combined()[0] == PASS
        assert shared._model.max_active == 1, "shared detector was entered concurrently"
        b.detector = detect.YoloDetector(model=detect.FakeYolo(
            lambda f: (_ for _ in ()).throw(RuntimeError("one camera's detector fails"))), warmup=False)
        wait(lambda: b.result()[0] == FAULT, "one detector faults")
        st, by = cs.combined()
        assert st == FAULT and "detector" in by["Camera 1"][0] and "Camera 0" not in by, (st, by)
        cs.stop()
    finally:
        me.open_capture, D.load_config = real_open, real_cfg


def _selftest_controls():
    """camera_controls: auto modes set before manual values, read-back reported (a driver may
    ignore a request), unknown names flagged; rotate turns the frame before it gets a seq, so a
    landscape camera mounted in portrait delivers a tall frame with seq still strictly increasing."""
    me = sys.modules[__name__]

    class Prop:                                # a driver that ignores focus but takes the rest
        def __init__(self):
            self.v, self.order = {}, []
        def set(self, p, val):
            self.order.append(p)
            if p != cv2.CAP_PROP_FOCUS:
                self.v[p] = val
        def get(self, p):
            return self.v.get(p, -1.0)

    cap = Prop()
    got = apply_controls(cap, {"focus": 40, "autofocus": 0, "exposure": -6, "rotate": 90, "zoom": 2})
    assert cap.order.index(cv2.CAP_PROP_AUTOFOCUS) < cap.order.index(cv2.CAP_PROP_FOCUS), cap.order
    assert got["focus"] == [40, -1.0] and got["exposure"] == [-6, -6.0], got
    assert got["zoom"] == [2, None] and "rotate" not in got, got
    f = np.zeros((36, 64, 3), np.uint8)
    assert orient(f, 90).shape == (64, 36, 3) and orient(f, 0) is f and orient(f, 45) is f

    class Cap:
        def isOpened(self):
            return True
        def get(self, *_):
            return 0
        def set(self, *_):
            pass
        def release(self):
            pass
        def read(self):
            time.sleep(0.004)
            return True, np.zeros((36, 64, 3), np.uint8)

    real_open, real_settings = me.open_capture, D.load_settings
    me.open_capture = lambda src, *_: Cap()
    D.load_settings = lambda: {"camera_controls": {"7": {"rotate": 90}}}
    try:
        c = Camera("portrait")
        c.start(7)
        t0 = time.monotonic()
        while len(c.recent) < 5:
            assert time.monotonic() - t0 < 3.0, "no frames from the rotated camera"
            time.sleep(0.01)
        c.stop()
        seqs = [fr.seq for fr in c.recent]
        assert c.rotate == 90 and c.frame_wh == (36, 64), (c.rotate, c.frame_wh)
        assert all(fr.image.shape[:2] == (64, 36) for fr in c.recent)
        assert seqs == sorted(set(seqs)), seqs
        c.start(8)                                     # no entry -> unrotated
        c.stop()
        assert c.rotate == 0
    finally:
        me.open_capture, D.load_settings = real_open, real_settings


def _selftest_verdict_overlay():
    """The operator overlay is ONE word in a colour: no boxes, no score bars, no per-class text."""
    c = Camera("t")
    base = np.zeros((300, 400, 3), np.uint8)
    for kind, defects, col in ((V.GOOD, (), C_PASS), (V.DEFECT, ("missing_cap",), C_FAIL), (V.FAULT, (), C_FAULT)):
        v = V.Verdict(kind, defects, "x" if kind == V.FAULT else "")
        out = c._industrial_overlay(base.copy(), v)
        assert tuple(int(x) for x in out[150, 2]) == tuple(col), (kind, out[150, 2])      # frame in the verdict colour
        assert tuple(int(x) for x in out[2, 200]) == tuple(col), kind                      # banner in the verdict colour
        assert not out[100:250, 40:360].any(), "the operator view drew something inside the picture"
    # no bottle / checking: neutral, no coloured frame
    out = c._industrial_overlay(base.copy(), V.Verdict(V.EMPTY))
    assert not out[150, 2].any() and tuple(int(x) for x in out[2, 200]) == c._VCOL[V.EMPTY]
    assert V.headline(V.DEFECT, ("missing_cap",)) == "DEFECT: Missing cap"
    # a stopped camera is FAULT, never a stale GOOD
    assert c.display_verdict().kind == V.FAULT


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

    _selftest_tri_state()
    _selftest_detector()
    _selftest_controls()
    _selftest_verdict_overlay()

    print(f"ok  ({len(cams)} camera(s) found on indices 0-1; "
          f"tri-state PASS/REJECT/FAULT + component-detector path + camera controls/rotation + "
          f"one-verdict operator overlay checked)")


if __name__ == "__main__":
    demo()
