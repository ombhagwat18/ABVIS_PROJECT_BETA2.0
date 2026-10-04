"""The machine cycle: PLC trigger -> per-bottle inspection -> decision -> time-based FIFO -> PLC command.

    X0 (photo-eye) -> ladder SET M2 -> PLCService Trigger
        -> Bottle "000123" (inspection_id) enters the FIFO, stamped with the trigger time
        -> Inspector: frames captured AFTER the trigger, from every line camera
           -> the selected AI task (classification / detection / segmentation)  [decision.py]
        -> Decision PASS / REJECT / FAULT
        -> PLC command at its deadline: PASS -> M0 at once; REJECT -> M1 at
           dispatch_at = scheduled_reject_time - T0 (the PLC times Y0 itself from M1)
        -> tracked until the bottle has passed the reject station (Y0 pulse observed for a REJECT)
        -> history + production log (CSV)

Software FIFO, not PLC bits. M0/M1/M2 are a one-bottle handshake; the queue of bottles between
the camera and the reject station lives here, keyed by inspection_id, each with its own deadline:

    travel_time           = inspection_to_reject_mm / conveyor_mm_s     (settings.json)
    scheduled_reject_time = t_trigger + travel_time
    dispatch_at (REJECT)  = scheduled_reject_time - T0                  (T0 = PLC travel-delay timer)

One thread, driven by deadlines: it blocks on "next trigger OR next deadline", never on a fixed
sleep. Python never times or drives Y0 -- the PLC does (T0 -> Y0 -> T1).

Every bottle ends with exactly ONE final result (PASS / REJECT / FAULT):
  * FAULT if it could not be inspected (camera fault, no frame, AI failure, no bottle seen), if the
    PLC command failed / was refused / was not acknowledged, if a REJECT missed its deadline, or
    if Y0 was never observed for it.
  * FAULT is physically REJECTED by default (settings "fault_action": "REJECT"): a bottle nobody
    could vouch for must not pass. The final result still says FAULT.
  * A REJECT whose deadline has already passed is NOT fired late (the cylinder would hit the wrong
    bottle or nothing): the handshake is answered with M0, the bottle is a FAULT and an alarm
    says it must be removed by hand.

Ladder limits this code works around rather than hides (docs/roadmap/PLC_COMMUNICATION.md section 0):
  * While M1 is ON (T0 + T1 after every REJECT) net 5 resets M2 on every scan, so a bottle reaching
    X0 in that window raises no trigger; the same happens if X0 rises while M2 is still ON. PLCService
    (watch_x0=True) reports every such X0 edge as BOTTLE_UNTRIGGERED and the bottle is recorded here as
    FAULT "NOT INSPECTED" -- not silently lost. (X0 is polled with the handshake bits, ~20-40 ms: a
    shorter X0 pulse cannot be seen, so this check can miss, never invent, a bottle.)
  * Only one REJECT can be in the PLC at a time (one T0). Throughput after a reject is limited to
    one bottle per T0 + T1.

    python machine_cycle.py     # self-test: FAKE PLC running the decoded ladder + fake cameras/models
"""
from __future__ import annotations

import collections
import csv
import itertools
import queue
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

import dataset as D
import decision as DEC
from infer import FAULT, PASS, REJECT

# FIFO / PLC status of a bottle
INSPECTING, SCHEDULED, SENT, IN_TRANSIT, REJECTING, DONE, NOT_INSPECTED = (
    "INSPECTING", "SCHEDULED", "SENT", "IN TRANSIT", "REJECTING", "DONE", "NOT INSPECTED")

LINE_DEFAULTS = {
    "line_task": "detection",            # decision.TASKS key
    "line_cameras": [],                  # camera sources (DirectShow indices) watching the station
    "line_capture_wh": [1920, 1080],     # requested capture mode for the line cameras
    "line_fourcc": "MJPG",
    "inspect_frames": 3,                 # frames per camera per bottle
    "inspect_window_s": 0.6,             # max wait for those frames after the trigger
    "capture_delay_s": 0.0,              # frames older than trigger + this are not used
    "inspection_to_reject_mm": 0.0,      # 0 = not measured: T0 is taken as the travel time
    "conveyor_mm_s": 0.0,
    "plc_t0_s": 1.5,                     # the ladder's T0 (travel delay) -- stated contract K15
    "plc_t1_s": 0.5,                     # the ladder's T1 (reject pulse) -- stated contract K5
    "reject_tolerance_s": 0.3,           # how late M1 may be sent and still hit the bottle
    "fault_action": "REJECT",            # what the PLC does with a FAULT bottle: REJECT | PASS
    "estop_device": "",                  # optional PLC input wired to the hardware E-stop status, e.g. "X3" ("" = none)
    "estop_active_high": False,          # False: the contact is NC, so the bit reads 0 when the E-stop is pressed
    "fault_latch_after": 3,              # this many consecutive FAULT bottles halt the line until an operator reset
    "decision_rules": {},
}


def line_settings(settings: dict) -> dict:
    s = dict(LINE_DEFAULTS)
    s.update({k: settings[k] for k in LINE_DEFAULTS if k in settings})
    return s


def travel_time(cfg: dict) -> tuple:
    """(seconds, measured?) from distance / speed; falls back to the PLC's T0 when not measured."""
    dist, speed = float(cfg.get("inspection_to_reject_mm") or 0), float(cfg.get("conveyor_mm_s") or 0)
    if dist > 0 and speed > 0:
        return dist / speed, True
    return float(cfg["plc_t0_s"]), False


# With distance/speed not measured, T0 IS the travel time. Above this it is almost certainly the old
# K150 (15 s) preset, not a travel time: on this machine's ~1460 mm belt at ~90 mm/s, inspection ->
# reject is a few seconds, and a 15 s T0 fires Y0 after the bottle has left the conveyor.
MAX_UNMEASURED_T0_S = 5.0


def timing_problem(cfg: dict) -> str | None:
    """Why these line timings cannot reject a bottle at the right place (None = plausible).

    The REJECT command is sent at trigger + travel - T0, so a T0 at or above the travel time makes
    every REJECT late before inspection even starts: each one would be answered PASS + FAULT."""
    t0 = float(cfg["plc_t0_s"])
    travel, measured = travel_time(cfg)
    if measured and t0 >= travel:
        return (f"T0 {t0:.2f} s >= travel {travel:.2f} s ({cfg['inspection_to_reject_mm']} mm at "
                f"{cfg['conveyor_mm_s']} mm/s): every REJECT would be late. Set the ladder's T0 below the "
                f"travel time (contract K15 = 1.5 s) and enter the same value here.")
    if not measured and t0 > MAX_UNMEASURED_T0_S:
        return (f"T0 {t0:.2f} s is used as the travel time because distance/speed are not measured, and "
                f"no inspection->reject distance on this belt takes that long. Measure the distance and "
                f"speed, or set T0 to the real travel time (ladder and here).")
    return None


@dataclass
class Bottle:
    inspection_id: str
    seq: int
    t_trigger: float                          # monotonic: when Python saw M2 (<= one poll after X0)
    wall: float                               # epoch seconds, display only
    trigger_id: int | None = None
    after_reconnect: bool = False
    status: str = INSPECTING
    decision: str | None = None               # what the AI concluded
    final: str | None = None                  # the one final result (differs from decision on PLC trouble)
    defects: list = field(default_factory=list)
    reason: str = ""
    confidence: float | None = None
    per_camera: dict = field(default_factory=dict)
    frames: int = 0
    infer_ms: float | None = None
    decided_mono: float | None = None
    command: str | None = None                # PASS | REJECT actually sent (or to be sent)
    travel_s: float | None = None
    scheduled_mono: float | None = None       # bottle at the reject station
    scheduled_wall: float | None = None
    dispatch_mono: float | None = None        # when the command is due
    sent_mono: float | None = None
    lateness_ms: float | None = None          # sent - dispatch_at (REJECT)
    plc_status: str = ""
    ack_ms: float | None = None
    y0_on_mono: float | None = None
    y0_off_mono: float | None = None
    y0_error_ms: float | None = None          # Y0 ON (as polled) - scheduled_reject_time
    note: str = ""
    thumb: object = None                      # small evidence image (not logged)

    def row(self) -> dict:
        """Flat, CSV/JSON-friendly copy (no image, wall times as text)."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("thumb", "per_camera")}
        out["defects"] = ";".join(self.defects)
        out["wall"] = _wall(self.wall)
        out["scheduled_wall"] = _wall(self.scheduled_wall)
        for k in ("t_trigger", "decided_mono", "scheduled_mono", "dispatch_mono", "sent_mono", "y0_on_mono",
                  "y0_off_mono"):
            v = out[k]
            out[k] = None if v is None else round(v - self.t_trigger, 4)      # relative to the trigger
        out["t_trigger"] = round(self.t_trigger, 4)
        for k in ("confidence", "infer_ms", "travel_s", "lateness_ms", "ack_ms", "y0_error_ms"):
            if out[k] is not None:
                out[k] = round(out[k], 3)
        out["per_camera"] = "; ".join(f"{k}={v[0]}:{v[1] if isinstance(v[1], str) else ','.join(v[1]) or 'ok'}"
                                      for k, v in self.per_camera.items())
        return out


def _wall(t):
    if t is None:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


# ------------------------------------------------------------------------------------- inspector
class Inspector:
    """Collects the frames each line camera captured after a trigger and runs the AI stages on them.

    cams: infer.Camera objects (capture threads already running). The models are run HERE, on demand,
    per bottle -- the camera threads only capture -- so every frame used is accounted for and the
    GPU is not shared with a free-running preview."""

    def __init__(self, cams, task, classifier=None, detector=None, segmenter=None, missing=None,
                 thresholds=None, frames=3, window_s=0.6, recipe=None):
        self.cams, self.task = list(cams), task
        self.classifier, self.detector, self.segmenter = classifier, detector, segmenter
        self.missing = dict(missing or {})      # stage -> why its model is unavailable
        self.thresholds = thresholds or (lambda: D.load_config().get("thresholds", {}))
        # the project's inspection recipe (config.json "inspection"); None = decision.default_recipe
        self.recipe = recipe or (lambda: D.load_config().get("inspection"))
        self.frames, self.window_s = max(1, int(frames)), float(window_s)
        self._wake = threading.Event()

    def models(self) -> dict:
        return {"classification": getattr(self.classifier, "stamp", None),
                "detection": getattr(self.detector, "model_id", None),
                "segmentation": getattr(self.segmenter, "model_id", None)}

    def collect(self, t_from: float):
        """{camera: (frames, fault)}; waits at most window_s for `frames` new frames per camera."""
        deadline = t_from + self.window_s
        got = {c.camera_id: {} for c in self.cams}
        faults: dict = {}
        while True:
            for c in self.cams:
                cid = c.camera_id
                if cid in faults:
                    continue
                if c.error or not c.alive:
                    faults[cid] = f"camera fault: {c.error or 'capture thread not running'}"
                    continue
                for f in c.frames_since(t_from):
                    # strictly after: monotonic() ticks every ~15.6 ms on Windows, so a frame grabbed just
                    # BEFORE the trigger can carry the same stamp -- and show the previous bottle
                    if f.ts > t_from and len(got[cid]) < self.frames:
                        got[cid].setdefault(f.seq, f)
            if all(cid in faults or len(v) >= self.frames for cid, v in got.items()):
                break
            if time.monotonic() >= deadline:
                break
            self._wake.wait(0.004)              # frames arrive every ~33 ms; this is a bounded wait for data
        out = {}
        for c in self.cams:
            cid = c.camera_id
            frames = [got[cid][k] for k in sorted(got[cid])]
            fault = faults.get(cid)
            if fault is None and not frames:
                fault = f"frame timeout: no frame within {self.window_s:g}s of the trigger"
            out[cid] = (frames, fault)
        return out

    def frame_evidence(self, cid, seq, ts, image) -> DEC.FrameEvidence:
        """Run this task's AI stages on ONE frame. A stage that is missing or raises becomes that
        stage's error on the evidence (decide() turns it into FAULT), never an exception."""
        stages = DEC.TASKS.get(self.task, ())
        fe = DEC.FrameEvidence(cid, seq, ts)
        if DEC.CLASSIFICATION in stages:
            if self.classifier is None:
                fe.cls_error = self.missing.get(DEC.CLASSIFICATION, "no classifier loaded")
            else:
                try:
                    fe.probs = self.classifier.predict(image)
                except Exception as e:                       # noqa: BLE001 - FAULT, not a crash
                    fe.cls_error = f"{type(e).__name__}: {e}"
        if DEC.DETECTION in stages:
            if self.detector is None:
                fe.det_error = self.missing.get(DEC.DETECTION, "no detector loaded")
            else:
                try:
                    fe.det = self.detector.detect(image, camera_id=cid, frame_seq=seq, frame_ts=ts)
                except Exception as e:                       # noqa: BLE001
                    fe.det_error = f"{type(e).__name__}: {e}"
        if DEC.SEGMENTATION in stages:
            if self.segmenter is None:
                fe.seg_error = self.missing.get(DEC.SEGMENTATION, "no segmentation model loaded")
            else:
                try:
                    fe.seg = self.segmenter.segment(image, camera_id=cid, frame_seq=seq, frame_ts=ts)
                except Exception as e:                       # noqa: BLE001
                    fe.seg_error = f"{type(e).__name__}: {e}"
        return fe

    def inspect(self, t_from: float):
        """(decision.Decision, evidence image or None, total inference ms, frames used)."""
        evid, ms, n, first = [], 0.0, 0, None
        thr = self.thresholds()
        for cid, (frames, fault) in self.collect(t_from).items():
            ce = DEC.CameraEvidence(cid, fault=fault)
            for f in frames:
                t0 = time.perf_counter()
                fe = self.frame_evidence(cid, f.seq, f.ts, f.image)
                ms += (time.perf_counter() - t0) * 1000
                n += 1
                ce.frames.append(fe)
                if first is None:
                    first = (f.image, fe)
            evid.append(ce)
        rules = dict(self._rules)
        recipe = self.recipe()
        if recipe:
            rules["recipe"] = recipe
        dec = DEC.decide(self.task, evid, thr, rules)
        return dec, (self._thumb(*first, dec) if first else None), ms, n

    _rules: dict = {}

    @staticmethod
    def _thumb(img, fe, dec, width=480):
        """Small evidence image: the first frame used, with detector boxes and the verdict."""
        h, w = img.shape[:2]
        s = width / max(1, w)
        out = cv2.resize(img, (width, max(1, round(h * s))))
        if fe.det is not None:
            col = {"bottle": (235, 160, 60), "cap": (60, 200, 255), "label": (120, 220, 120)}
            for d in fe.det.detections:
                cv2.rectangle(out, (round(d.x1 * s), round(d.y1 * s)), (round(d.x2 * s), round(d.y2 * s)),
                              col.get(d.class_name, (180, 180, 180)), 2)
        if fe.seg is not None:
            for sg in fe.seg.segments:
                cv2.polylines(out, [np.round(sg.polygon * s).astype(np.int32)], True, (255, 0, 255), 2)
        colour = {PASS: (60, 160, 40), REJECT: (40, 40, 210)}.get(dec.state, (0, 140, 230))
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), colour, -1)
        cv2.putText(out, f"{dec.state}  {dec.defect_text}"[:60], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        return out


# ---------------------------------------------------------------------------------- machine cycle
class MachineCycle:
    """Owns the per-bottle loop and the FIFO. Talks to the PLC only through PLCService."""

    def __init__(self, plc, inspector: Inspector, cfg: dict, log_dir: Path | None = None, clock=time.monotonic):
        self.plc, self.inspector, self.cfg = plc, inspector, dict(cfg)
        self.inspector._rules = dict(self.cfg.get("decision_rules") or {})
        self.travel_s, self.travel_measured = travel_time(self.cfg)
        self.clock = clock
        self.log_dir = log_dir
        self._log_path: Path | None = None
        self._lock = threading.Lock()
        self._fifo: "collections.deque[Bottle]" = collections.deque()
        self._history: "collections.deque[Bottle]" = collections.deque(maxlen=500)
        self._events: "queue.Queue" = queue.Queue()
        self._ids = itertools.count(1)
        self.counts = {"total": 0, PASS: 0, REJECT: 0, FAULT: 0, "not_inspected": 0, "missed_reject": 0}
        self.alarms: "collections.deque" = collections.deque(maxlen=200)    # (wall, text)
        self.last: Bottle | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._listening = False
        self.halted: str | None = None           # latched reason; set => no PLC command is sent until reset()
        self._fault_run = 0
        self._estop_next = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not self._listening:
            self.plc.add_listener(self._on_plc_event)
            self._listening = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="machine-cycle", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        if self._listening:
            try:
                self.plc._listeners.remove(self._on_plc_event)
            except ValueError:
                pass
            self._listening = False

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ safety latch
    # This is the SOFTWARE layer only. The hardware E-stop must cut the actuator power by itself;
    # software halt just stops this program from answering the PLC and tells the operator why.
    def halt(self, reason: str):
        """Latch a halt. Idempotent; the first reason wins. Scheduled REJECTs are cancelled, not fired."""
        with self._lock:
            if self.halted:
                return
            self.halted = reason
            pending = [b for b in self._fifo if b.status == SCHEDULED]
        self._alarm(f"LINE HALTED: {reason}")
        for b in pending:
            b.plc_status = "HALTED before the REJECT was sent: remove by hand"
            self._finish(b, FAULT)

    def reset(self) -> bool:
        """Operator reset. Refused while the hardware E-stop input still reads pressed."""
        if self._estop_pressed():
            self._alarm("reset refused: hardware E-stop input is still active")
            return False
        with self._lock:
            self.halted = None
            self._fault_run = 0
        self._alarm("halt cleared by operator")
        return True

    def _estop_pressed(self) -> bool:
        dev = str(self.cfg.get("estop_device") or "").strip()
        if not dev:
            return False
        try:
            val = bool(self.plc.sample(bits=[dev])[dev.upper()])
        except Exception:                                            # noqa: BLE001 - cannot read it => treat as pressed
            return True
        return val if self.cfg.get("estop_active_high") else not val

    def _watch_estop(self):
        if not str(self.cfg.get("estop_device") or "").strip() or self.clock() < self._estop_next:
            return
        self._estop_next = self.clock() + 0.25
        if self._estop_pressed():
            self.halt(f"hardware E-stop input {self.cfg['estop_device']} active (or unreadable)")

    # ------------------------------------------------------------------ views (any thread)
    def snapshot(self) -> dict:
        with self._lock:
            fifo = [b for b in self._fifo]
            hist = list(self._history)[-60:]
            return {"counts": dict(self.counts), "queue": len(fifo), "fifo": fifo, "history": hist,
                    "alarms": list(self.alarms)[-20:], "last": self.last, "running": self.running,
                    "error": self.error, "halted": self.halted, "travel_s": self.travel_s, "travel_measured": self.travel_measured}

    # ------------------------------------------------------------------ PLC events (service thread)
    def _on_plc_event(self, ev):
        if (ev.event == "DEVICE_CHANGE" and ev.device in ("Y0", "M1")) or ev.event == "BOTTLE_UNTRIGGERED":
            self._events.put(ev)

    # ------------------------------------------------------------------ the loop
    def _run(self):
        try:
            while not self._stop.is_set():
                wait = max(0.0, min(0.05, self._next_deadline() - self.clock()))
                self._watch_estop()
                trig = self.plc.wait_for_trigger(wait)
                if trig is not None:
                    self._on_trigger(trig)
                self._drain_events()
                self._due()
        except Exception as e:                                       # noqa: BLE001 - shown, not swallowed
            import traceback
            traceback.print_exc()
            self.error = f"machine cycle crashed: {type(e).__name__}: {e}"
            self._alarm(self.error)
            self.halted = self.halted or self.error

    def _next_deadline(self) -> float:
        now = self.clock()
        ds = [now + 0.05]
        with self._lock:
            for b in self._fifo:
                if b.status == SCHEDULED and b.dispatch_mono is not None:
                    ds.append(b.dispatch_mono)
        return min(ds)

    # ------------------------------------------------------------------ trigger -> inspection
    def _new_bottle(self, t: float, wall: float, **kw) -> Bottle:
        n = next(self._ids)
        b = Bottle(f"{n:06d}", n, t, wall, **kw)
        b.travel_s = self.travel_s
        b.scheduled_mono = t + self.travel_s
        b.scheduled_wall = wall + self.travel_s
        with self._lock:
            self._fifo.append(b)
        return b

    def _on_trigger(self, trig):
        b = self._new_bottle(trig.seen_mono, trig.seen_wall, trigger_id=trig.id, after_reconnect=trig.after_reconnect)
        if self.halted:                                              # never answer while halted
            b.decision, b.command, b.reason = FAULT, "", f"line halted: {self.halted}"
            b.plc_status = "NOT ANSWERED (halted): remove bottle by hand"
            self._alarm(f"bottle {b.inspection_id}: {b.plc_status}")
            self._finish(b, FAULT)
            return
        if trig.after_reconnect:
            b.note = "trigger seen right after (re)connect: M2 may be stale"
        t_from = trig.seen_mono + float(self.cfg["capture_delay_s"])
        try:
            dec, thumb, ms, n = self.inspector.inspect(t_from)
        except Exception as e:                                       # noqa: BLE001 - FAULT, never a crash
            dec = DEC.Decision(FAULT, [], f"inspection crashed: {type(e).__name__}: {e}", task=self.inspector.task)
            thumb, ms, n = None, None, 0
        b.decision, b.defects, b.reason, b.confidence = dec.state, list(dec.defects), dec.reason, dec.confidence
        b.per_camera, b.thumb, b.infer_ms, b.frames = dec.per_camera, thumb, ms, n
        b.decided_mono = self.clock()
        with self._lock:
            self.last = b
        cmd = {PASS: PASS, REJECT: REJECT}.get(dec.state, str(self.cfg["fault_action"]).upper())
        b.command = cmd if cmd in (PASS, REJECT) else REJECT
        if b.command == REJECT:
            b.dispatch_mono = b.scheduled_mono - float(self.cfg["plc_t0_s"])
            if b.dispatch_mono > self.clock():
                b.status = SCHEDULED                                 # sent by _due() at its deadline
                return
        self._send(b)

    def _send(self, b: Bottle):
        now = self.clock()
        cmd = b.command
        if cmd == REJECT and self.travel_measured:
            b.lateness_ms = (now - b.dispatch_mono) * 1000
            if now - b.dispatch_mono > float(self.cfg["reject_tolerance_s"]):
                # Too late to hit this bottle: do not fire Y0 at whatever is in front of the cylinder now.
                cmd = PASS
                b.note = (f"REJECT deadline missed by {b.lateness_ms:.0f} ms: NOT ejected, remove bottle "
                          f"{b.inspection_id} by hand")
                self.counts["missed_reject"] += 1
                self._alarm(f"bottle {b.inspection_id}: {b.note}")
        elif cmd == REJECT:
            b.lateness_ms = (now - b.dispatch_mono) * 1000          # informational: travel not measured
        b.sent_mono = now
        b.status = SENT
        res = self.plc.submit_result(b.trigger_id, cmd)
        b.ack_ms = res.ack_ms
        if res.ok:
            if cmd == REJECT:
                b.status, b.plc_status = REJECTING, "M1 ACKED, T0 running"
            else:
                b.status, b.plc_status = IN_TRANSIT, "M0 ACKED"
        else:
            b.status = DONE
            b.plc_status = f"{cmd} {res.status}: {res.detail}"[:160]
            self._alarm(f"bottle {b.inspection_id}: PLC {cmd} {res.status} - {res.detail}"[:200])
        if b.note.startswith("REJECT deadline missed") or not res.ok:
            self._finish(b, FAULT)
        elif cmd == PASS:
            pass                                                     # leaves the FIFO once past the reject station

    # ------------------------------------------------------------------ PLC observations
    def _drain_events(self):
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                return
            on = ev.ack == "ON"
            if ev.event == "BOTTLE_UNTRIGGERED":
                self._not_inspected(ev)
            elif ev.device == "Y0":
                with self._lock:
                    rej = [b for b in self._fifo if b.status == REJECTING]
                if not rej:
                    if on:
                        self._alarm("Y0 (reject) turned ON with no REJECT pending in the FIFO")
                    continue
                b = rej[0]                                           # one T0 in the ladder: the oldest reject
                if on and b.y0_on_mono is None:
                    b.y0_on_mono = ev.ts
                    b.y0_error_ms = (ev.ts - b.scheduled_mono) * 1000
                    b.plc_status = "Y0 ON (reject firing)"
                elif not on and b.y0_on_mono is not None:
                    b.y0_off_mono = ev.ts
                    b.plc_status = f"Y0 pulse {(ev.ts - b.y0_on_mono) * 1000:.0f} ms (as polled)"
                    self._finish(b, b.decision if b.decision in (REJECT, FAULT) else REJECT)
            elif ev.device == "M1" and not on:
                with self._lock:
                    rej = [b for b in self._fifo if b.status == REJECTING]
                if rej and rej[0].y0_on_mono is None:                # cycle ended and Y0 was never seen
                    b = rej[0]
                    b.plc_status = "M1 cleared but Y0 was never observed ON"
                    self._alarm(f"bottle {b.inspection_id}: reject cycle ended without an observed Y0 pulse")
                    self._finish(b, FAULT)

    def _not_inspected(self, ev):
        """X0 rose but the PLC could not raise a trigger: the bottle passes uninspected and unrejected."""
        b = self._new_bottle(ev.ts, ev.wall, status=NOT_INSPECTED)
        b.decision, b.command = FAULT, None
        b.reason = f"bottle sensed at X0 but no inspection trigger: {ev.error}"
        b.plc_status = "no trigger: not inspected, not rejected"
        self.counts["not_inspected"] += 1
        self._alarm(f"bottle {b.inspection_id}: NOT INSPECTED - {ev.error}")
        self._finish(b, FAULT)

    def _due(self):
        now = self.clock()
        with self._lock:
            due = [b for b in self._fifo if b.status == SCHEDULED and b.dispatch_mono <= now]
            passed = [b for b in self._fifo if b.status == IN_TRANSIT and now >= b.scheduled_mono]
            stuck = [b for b in self._fifo if b.status == REJECTING and b.sent_mono is not None
                     and now - b.sent_mono > float(self.cfg["plc_t0_s"]) + float(self.cfg["plc_t1_s"]) + 3.0]
        for b in due:
            self._send(b)
        for b in passed:                                              # PASS bottle is past the reject station
            self._finish(b, b.decision if b.decision == PASS else FAULT)
        for b in stuck:
            b.plc_status = "no Y0 / M1 change observed after the REJECT command"
            self._alarm(f"bottle {b.inspection_id}: {b.plc_status}")
            self._finish(b, FAULT)

    # ------------------------------------------------------------------ completion
    def _finish(self, b: Bottle, final: str):
        b.final = final
        b.status = DONE if b.status != NOT_INSPECTED else NOT_INSPECTED
        with self._lock:
            try:
                self._fifo.remove(b)
            except ValueError:
                return                                                # already finished
            self._history.append(b)
            self.counts["total"] += 1
            self.counts[final] += 1
        self._log(b)
        self._fault_run = self._fault_run + 1 if final == FAULT else 0
        limit = int(self.cfg.get("fault_latch_after") or 0)
        if limit and self._fault_run >= limit and not self.halted:
            self.halt(f"{self._fault_run} consecutive FAULT bottles")

    def _alarm(self, text: str):
        with self._lock:
            self.alarms.append((time.time(), text))

    def _log(self, b: Bottle):
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{time.strftime('%Y%m%d')}.csv"
            row = b.row()
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            self._log_path = path
        except OSError as e:
            self._alarm(f"production log not written: {e}")


# ------------------------------------------------------------------------------------- self-test
def demo():
    """FAKE PLC running the decoded ladder (plc.test_simulation.FakeLadder, timers scaled) + two
    fake cameras + a fake YOLO detector whose output follows the 'bottle in front of the camera'.
    Not hardware evidence: it proves the cycle, FIFO, timing and fault logic."""
    import shutil
    import tempfile

    import detect
    import infer
    from plc import PLCService
    from plc.test_simulation import FakeLadder, FakePLC, _client

    # ---- timing sanity (pure): T0 must be shorter than the travel it delays
    T = lambda **k: line_settings({"plc_t0_s": 1.5, "plc_t1_s": 0.5, **k})          # noqa: E731
    assert timing_problem(T()) is None                                      # unmeasured, contract K15
    assert "not measured" in timing_problem(T(plc_t0_s=15.0))               # the saved K150 ladder
    assert timing_problem(T(inspection_to_reject_mm=400, conveyor_mm_s=90)) is None     # 4.4 s travel
    assert "late" in timing_problem(T(plc_t0_s=15.0, inspection_to_reject_mm=400, conveyor_mm_s=90))
    assert "late" in timing_problem(T(plc_t0_s=1.5, inspection_to_reject_mm=100, conveyor_mm_s=100))

    scale = 0.2                                     # 100 ms timer base -> 20 ms: T0 = 0.3 s, T1 = 0.1 s
    t0_s, t1_s = 15 * 0.1 * scale, 5 * 0.1 * scale
    scene = {"kind": "good"}

    class Cap:
        def __init__(self):
            self.n = 0
        def isOpened(self):
            return True
        def get(self, *_):
            return 0
        def set(self, *_):
            return True
        def release(self):
            pass
        def read(self):
            time.sleep(0.01)
            self.n += 1
            if scene.get("dead"):
                return False, None
            img = np.zeros((300, 200, 3), np.uint8)
            img[0, 0, 0] = {"good": 1, "nocap": 2, "nolabel": 3, "empty": 4}[scene["kind"]]
            return True, img

    def yolo(frame):
        kind = {1: "good", 2: "nocap", 3: "nolabel", 4: "empty"}[int(frame[0, 0, 0])]
        if scene.get("det_boom"):
            raise RuntimeError("cuda error")
        boxes = {"bottle": ([60, 40, 140, 290], 0.95), "cap": ([80, 25, 120, 60], 0.9),
                 "label": ([65, 120, 135, 220], 0.88)}
        if kind == "nocap":
            boxes.pop("cap")
        if kind == "nolabel":
            boxes.pop("label")
        if kind == "empty":
            boxes = {}
        names = list(boxes)
        return ([boxes[n][0] for n in names] or np.zeros((0, 4)), [boxes[n][1] for n in names],
                [detect.CLASS_NAMES.index(n) for n in names])

    real_open = infer.open_capture
    infer.open_capture = lambda src, *_: Cap()
    fake = FakePLC()
    lad = FakeLadder(fake, scale=scale)
    svc = PLCService(_client(fake, timeout=0.5), poll_s=0.01, status_period_s=0.03, watch_x0=True)
    svc.start(); svc.connect()
    cams = []
    for i in range(2):
        c = infer.Camera(f"cam{i}")
        c.start(i)
        cams.append(c)
    det = detect.YoloDetector(model=detect.FakeYolo(yolo), warmup=False)
    logdir = Path(tempfile.mkdtemp(prefix="mc_"))
    cfg = line_settings({"plc_t0_s": t0_s, "plc_t1_s": t1_s, "inspect_frames": 2, "inspect_window_s": 0.5,
                         "fault_latch_after": 0})
    mc = MachineCycle(svc, Inspector(cams, "detection", detector=det, frames=2, window_s=0.5), cfg, logdir)
    mc.start()

    def wait(cond, what, timeout=5.0):
        t_end = time.monotonic() + timeout
        while not cond():
            assert time.monotonic() < t_end, f"timed out waiting for: {what}"
            time.sleep(0.01)

    def bottle(kind, settle=None):
        """One bottle: put it in front of the cameras, break the X0 beam, wait for it to finish."""
        scene["kind"] = kind
        n0 = mc.snapshot()["counts"]["total"]
        lad.trigger()
        wait(lambda: mc.snapshot()["counts"]["total"] > n0, f"bottle {kind} to finish", 6.0)
        return mc.snapshot()["history"][-1]

    try:
        wait(lambda: all(c.latest_frame() is not None for c in cams), "cameras")
        # ---- the required sequence: PASS -> REJECT -> PASS -> REJECT -> REJECT -> PASS
        seq = ["good", "nocap", "good", "nolabel", "nocap", "good"]
        got = [bottle(k) for k in seq]
        assert [b.final for b in got] == [PASS, REJECT, PASS, REJECT, REJECT, PASS], [b.final for b in got]
        assert [b.inspection_id for b in got] == [f"{i:06d}" for i in range(1, 7)]
        assert got[1].defects == ["missing_cap"] and got[3].defects == ["missing_label"], got[1].defects
        assert len({b.trigger_id for b in got}) == 6, "a trigger was reused"
        cmd_writes = [a for a, v in fake.writes if v == 1]
        from plc import address_map as AM
        assert cmd_writes == [AM.address_of(x) for x in ("M0", "M1", "M0", "M1", "M1", "M0")], cmd_writes
        assert fake.regs[AM.address_of("C0")] == 3 and fake.regs[AM.address_of("C1")] == 3
        assert [v for _, v in lad.y0_log] == [1, 0] * 3, lad.y0_log
        for b in got:
            assert b.frames == 4 and b.decided_mono - b.t_trigger < 0.5, (b.frames, b.decided_mono - b.t_trigger)
            if b.final == REJECT:
                assert b.y0_on_mono and b.y0_off_mono and b.plc_status.startswith("Y0 pulse"), b.plc_status
                # T0 (not measured travel) is the travel time: Y0 ~ T0 after the command, within polling
                assert abs((b.y0_on_mono - b.sent_mono) - t0_s) < 0.12, b.y0_on_mono - b.sent_mono
            else:
                assert b.y0_on_mono is None and b.plc_status == "M0 ACKED"
        assert mc.snapshot()["queue"] == 0 and mc.counts[PASS] == 3 and mc.counts[REJECT] == 3

        # ---- a bottle arriving during a reject cycle raises no trigger: recorded, not lost
        scene["kind"] = "nocap"
        lad.trigger(hold_s=0.06)                                    # reject #4
        wait(lambda: any(b.status == REJECTING for b in mc.snapshot()["fifo"]), "reject in progress")
        lad.trigger(hold_s=0.08)                                    # next bottle while M1 is held
        wait(lambda: mc.counts["not_inspected"] == 1, "masked bottle recorded", 3.0)
        wait(lambda: mc.snapshot()["queue"] == 0, "reject #4 done")
        h = mc.snapshot()["history"]
        miss = next(b for b in h if b.status == NOT_INSPECTED)
        assert miss.final == FAULT and "net 5" in miss.reason and miss.trigger_id is None, miss.reason
        assert sum(1 for b in h if b.status == NOT_INSPECTED) == 1

        # ---- no bottle in view -> FAULT -> physically rejected (fault_action), final stays FAULT
        b = bottle("empty")
        assert b.final == FAULT and b.command == REJECT and "no bottle" in b.reason, (b.final, b.reason)
        # ---- AI runtime failure -> FAULT
        scene["det_boom"] = True
        b = bottle("good")
        assert b.final == FAULT and "detection failed" in b.reason, b.reason
        scene["det_boom"] = False
        # ---- one camera dead -> FAULT (camera failure)
        cams[1].stop()
        b = bottle("good")
        assert b.final == FAULT and "camera" in b.reason, b.reason
        cams[1].start(1)
        wait(lambda: cams[1].latest_frame() is not None, "camera back")
        assert bottle("good").final == PASS

        # ---- measured travel: REJECT is held back and sent at scheduled_reject_time - T0
        mc.stop()
        cfg2 = dict(cfg, inspection_to_reject_mm=120.0, conveyor_mm_s=200.0)       # 0.6 s travel
        mc2 = MachineCycle(svc, mc.inspector, cfg2, logdir)
        mc2._ids = itertools.count(100)
        mc2.start()
        mc_ref = mc
        mc = mc2
        scene["kind"] = "nocap"
        b = bottle("nocap")
        assert b.final == REJECT and abs(b.travel_s - 0.6) < 1e-9
        assert abs((b.sent_mono - b.t_trigger) - (0.6 - t0_s)) < 0.06, b.sent_mono - b.t_trigger
        assert abs(b.y0_error_ms) < 120, b.y0_error_ms        # Y0 lands on the scheduled time, within polling
        # ---- a REJECT decided after its deadline is not fired late
        slow = mc2.inspector.inspect
        mc2.inspector.inspect = lambda t: (time.sleep(0.75), slow(t))[1]          # inspection slower than travel
        b = bottle("nocap")
        assert b.final == FAULT and "deadline missed" in b.note and b.command == REJECT, (b.final, b.note)
        assert b.y0_on_mono is None and mc2.counts["missed_reject"] == 1
        mc2.inspector.inspect = slow
        mc2.stop()
        # ---- safety latch: halted line never answers the PLC; reset clears it
        mc3 = MachineCycle(svc, mc.inspector, dict(cfg, fault_latch_after=2), logdir)
        mc3._ids = itertools.count(300)
        mc3.start()
        mc = mc3
        n_w = len(fake.writes)
        scene["det_boom"] = True
        assert bottle("good").final == FAULT and not mc3.halted
        assert bottle("good").final == FAULT
        wait(lambda: mc3.halted, "latch after consecutive FAULTs", 2.0)
        assert "consecutive FAULT" in mc3.halted, mc3.halted
        scene["det_boom"] = False
        w_before = len(fake.writes)
        b = bottle("good")                                          # trigger while halted
        assert b.final == FAULT and "NOT ANSWERED" in b.plc_status, b.plc_status
        assert len(fake.writes) == w_before, "wrote to the PLC while halted"
        assert mc3.snapshot()["halted"]
        assert mc3.reset() and not mc3.halted
        mc3.halt("operator STOP")
        assert mc3.halted == "operator STOP" and mc3.reset() and not mc3.halted
        mc3.stop()
        # hardware E-stop input: NC contact reads 0 when pressed; an unreadable input counts as pressed
        class _P:
            v, boom = 1, False
            def sample(self, bits=(), words=()):
                if self.boom:
                    raise OSError("link down")
                return {bits[0]: self.v}
        p = _P()
        mc4 = MachineCycle(p, mc.inspector, dict(cfg, estop_device="X3", fault_latch_after=0), logdir)
        mc4._watch_estop(); assert not mc4.halted
        p.v = 0; mc4._estop_next = 0; mc4._watch_estop()
        assert mc4.halted and "E-stop" in mc4.halted and not mc4.reset()      # cannot reset while pressed
        p.v = 1; assert mc4.reset()
        p.boom = True; mc4._estop_next = 0; mc4._watch_estop()
        assert mc4.halted, "unreadable E-stop input must halt"
        assert any(fake.writes) and not any(a in (AM.address_of("Y0"), AM.address_of("Y1")) for a, _ in fake.writes)
        rows = list(csv.DictReader((logdir / f"{time.strftime('%Y%m%d')}.csv").open(encoding="utf-8")))
        assert len(rows) == mc_ref.counts["total"] + mc2.counts["total"] + mc3.counts["total"], len(rows)
        assert {r["final"] for r in rows} == {PASS, REJECT, FAULT}
        print(f"ok  machine cycle (FAKE PLC + decoded ladder emulation, fake cameras/detector): "
              f"PASS,REJECT,PASS,REJECT,REJECT,PASS with unique ids + one final each, M0/M1/C0/C1/Y0 checked, "
              f"masked bottle -> NOT INSPECTED, no-bottle/AI/camera failures -> FAULT (rejected), "
              f"deadline scheduling + missed deadline not fired, {len(rows)} rows logged")
    finally:
        mc.stop()
        for c in cams:
            c.stop()
        svc.stop(); lad.stop(); fake.close()
        infer.open_capture = real_open
        shutil.rmtree(logdir, ignore_errors=True)


def bench_inspect(label: str, sources, task: str = "classification+detection", frames: int = 3,
                  size=(1920, 1080), fourcc: str = "MJPG", out_root=Path("captures")) -> dict:
    """Bench test with real cameras and NO PLC: what would the line decide about the bottle in front
    of the cameras right now?

    sources: camera indices (captured one at a time -- two EMEETs on one USB 2.0 hub cannot both
    stream 1080p) or image file paths. The same Inspector.frame_evidence + decision.decide the line
    uses; the same models the Production tab loads (active classifier, provenance-checked YOLO).
    `label` is what the operator says is in front of the camera (e.g. GOOD / MISSING_CAP); it is
    only recorded beside the result, never used by it. Writes raw frames, overlays and result.json
    to captures/<label>_<stamp>/.
    """
    import json
    import detect
    import infer
    import segment
    settings = D.load_settings()
    stages = DEC.TASKS[task]
    models, missing = {}, {}
    if DEC.CLASSIFICATION in stages:
        stamp = D.load_config().get("active_model")
        try:
            models["classifier"] = infer.Model(stamp) if stamp else None
            if not stamp:
                missing[DEC.CLASSIFICATION] = f"project {D.PROJECT!r} has no active classifier"
        except Exception as e:                                       # noqa: BLE001
            missing[DEC.CLASSIFICATION] = f"{type(e).__name__}: {e}"
    if DEC.DETECTION in stages:
        try:
            models["detector"] = detect.YoloDetector(
                require=DEC.recipe_classes(DEC.recipe_of({**DEC.RULES, "recipe": D.load_config().get("inspection")})),
                weights=settings.get("detector_weights") or None,
                                                     conf=float(settings.get("detector_conf", detect.DEV_CONF)))
        except Exception as e:                                       # noqa: BLE001
            missing[DEC.DETECTION] = str(e)
    if DEC.SEGMENTATION in stages:
        try:
            models["segmenter"] = segment.YoloSegmenter(weights=settings.get("segmenter_weights") or None)
        except Exception as e:                                       # noqa: BLE001
            missing[DEC.SEGMENTATION] = str(e)
    insp = Inspector([], task, missing=missing, **models)
    out = Path(out_root) / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    names = infer.camera_names()
    evid, rec = [], {"label": label, "task": task, "project": D.PROJECT, "models": insp.models(),
                     "missing": missing, "rules": {**DEC.RULES, **settings.get("decision_rules", {}),
                                                   **({"recipe": D.load_config()["inspection"]}
                                                      if D.load_config().get("inspection") else {})},
                     "cameras": {}}
    for src in sources:
        is_cam = str(src).isdigit()
        cid = f"cam{src}" if is_cam else Path(src).stem
        info = {"source": str(src)}
        imgs, ce = [], DEC.CameraEvidence(cid)
        if is_cam:
            i = int(src)
            info["name"] = names[i] if i < len(names) else ""
            cap = infer.open_capture(i, size, fourcc)
            rot = infer.camera_controls(i).get("rotate") or 0       # same orientation as the line
            info["rotate"], info["controls"] = rot, infer.applied_controls.get(str(i))
            try:
                for _ in range(15):                                  # let auto-exposure settle
                    cap.read()
                for _ in range(frames):
                    ok, img = cap.read()
                    if ok and img is not None:
                        imgs.append(infer.orient(img, rot))
                    time.sleep(0.1)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(src))
            imgs = [img] if img is not None else []
        if not imgs:
            ce.fault = "no frame captured" if is_cam else f"cannot read {src}"
        info["frames"] = []
        for k, img in enumerate(imgs, 1):
            info["size"] = f"{img.shape[1]}x{img.shape[0]}"
            cv2.imwrite(str(out / f"{cid}_f{k}.png"), img)
            t0 = time.perf_counter()
            fe = insp.frame_evidence(cid, k, time.monotonic(), img)
            fr = {"ms": round((time.perf_counter() - t0) * 1000, 1)}
            if fe.probs is not None:
                fr["probs"] = {d: round(p, 3) for d, p in fe.probs.items()}
            if fe.det is not None:
                fr["detections"] = [[d.class_name, round(d.confidence, 3), [round(d.x1), round(d.y1),
                                     round(d.x2), round(d.y2)]] for d in fe.det.detections]
                found, _ = DEC.detection_findings(fe.det, rec["rules"])
                fr["detection_findings"] = "no bottle found" if found is None else found
            for k2 in ("cls_error", "det_error", "seg_error"):
                if getattr(fe, k2):
                    fr[k2] = getattr(fe, k2)
            info["frames"].append(fr)
            ce.frames.append(fe)
        evid.append(ce)
        rec["cameras"][cid] = info
    thr = D.load_config().get("thresholds", {})
    dec = DEC.decide(task, evid, thr, rec["rules"])
    for ce in evid:                                                  # overlays: bottle-level verdict on every frame
        for k, fe in enumerate(ce.frames, 1):
            img = cv2.imread(str(out / f"{ce.camera}_f{k}.png"))
            cv2.imwrite(str(out / f"{ce.camera}_f{k}_overlay.jpg"), Inspector._thumb(img, fe, dec, width=960))
    rec["decision"] = {"state": dec.state, "defects": dec.defects, "reason": dec.reason,
                       "per_camera": {k: list(v) for k, v in dec.per_camera.items()}}
    (out / "result.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"{label}: {dec.state} {dec.defects} -- {dec.reason}\n  -> {out}")
    return rec


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        # python machine_cycle.py --bench GOOD 2 3 [--task detection] [--frames 3]
        # python machine_cycle.py --bench GOOD path/to/a.png path/to/b.png
        a = sys.argv[sys.argv.index("--bench") + 1:]
        opt = {}
        for flag in ("--task", "--frames"):
            if flag in a:
                i = a.index(flag)
                opt[flag[2:]] = a[i + 1]
                del a[i:i + 2]
        bench_inspect(a[0], a[1:], task=opt.get("task", "classification+detection"),
                      frames=int(opt.get("frames", 3)))
    else:
        demo()
