"""One stable verdict per bottle, in plain words: GOOD, DEFECT (with the defect's name), CHECKING, NO BOTTLE, FAULT.

Why: a per-frame classifier flickers. At ~15 scores a second a good bottle with one noisy frame flashes REJECT, and
a screen full of boxes and probability bars tells an operator nothing. An inspection station shows ONE answer per
bottle, the moment it is sure, and keeps it until the bottle has left.

    bottle arrives -> settle (ignore the first frames: it is still moving into view)
                   -> collect `min_frames` valid frames
                   -> a defect counts only if it is in >= `vote` of them
                   -> GOOD or DEFECT, LATCHED (no flicker) until the bottle has been absent `absent_frames`
    nothing valid for `fault_frames` frames (camera / model / detector failure) -> FAULT (never GOOD)

Presence comes from the component detector (a bottle box). With no detector presence is unknown (None): the
verdict then follows a rolling window instead of latching, because there is no "bottle left" signal to unlatch on.

This module only DISPLAYS what the classifier / detector found. It does not command the PLC: the machine line's
decision stays in decision.py + machine_cycle.py.

    python verdict.py     # self-test
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

EMPTY, CHECKING, GOOD, DEFECT, FAULT = "NO BOTTLE", "CHECKING", "GOOD", "DEFECT", "FAULT"

DEFECT_NAMES = {
    "damaged_bottle": "Damaged bottle", "damaged_label": "Damaged label", "missing_cap": "Missing cap",
    "missing_label": "Missing label", "skewed_bottle": "Skewed bottle", "skewed_label": "Skewed label",
    "tilt_cap": "Tilted cap", "water_level": "Wrong water level", "cap_misplaced": "Cap misplaced",
    "label_area_low": "Label too small", "label_damaged": "Damaged label",
}


def pretty(defect: str) -> str:
    """'missing_cap' -> 'Missing cap' (unknown names are made readable, never dropped)."""
    return DEFECT_NAMES.get(defect) or defect.replace("_", " ").capitalize()


def headline(kind: str, defects=(), reason: str = "") -> str:
    """The one line an operator reads: GOOD / DEFECT: Missing cap / CHECKING... / NO BOTTLE / FAULT: why."""
    if kind == GOOD:
        return "GOOD"
    if kind == DEFECT:
        names = [pretty(d) for d in defects]
        return "DEFECT: " + (", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "") if names else "see details")
    if kind == CHECKING:
        return "CHECKING..."
    if kind == FAULT:
        return "FAULT" + (f": {reason}" if reason else "")
    return "NO BOTTLE"


def shown_result(final: str | None) -> str:
    """PLC-facing PASS / REJECT / FAULT -> the words on screen."""
    return {"PASS": GOOD, "REJECT": DEFECT, "FAULT": FAULT}.get(final or "", final or "--")


@dataclass
class Verdict:
    kind: str = EMPTY
    defects: tuple = ()
    reason: str = ""
    frames: int = 0                  # valid frames this verdict is based on
    since: float = 0.0               # monotonic time the verdict was reached
    latched: bool = False

    @property
    def text(self) -> str:
        return headline(self.kind, self.defects, self.reason)


@dataclass
class _Cfg:
    settle_frames: int = 2           # frames ignored after a bottle appears
    min_frames: int = 6              # valid frames needed for a verdict (~0.4 s at 15 Hz)
    vote: float = 0.6                # fraction of those frames a defect must appear in
    absent_frames: int = 8           # frames without a bottle before the verdict is cleared
    fault_frames: int = 4            # consecutive invalid frames before FAULT
    window: int = 10                 # rolling window when presence is unknown


class VerdictTracker:
    """Feed one call per scored frame; read .current() from any thread."""

    def __init__(self, clock=time.monotonic, **cfg):
        self.cfg = _Cfg(**cfg)
        self.clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._frames: list = []                   # defect sets of the valid frames of this bottle
            self._seen = 0                            # frames since the bottle appeared (incl. settle)
            self._absent = 0
            self._invalid = 0
            self._roll: deque = deque(maxlen=self.cfg.window)
            self._v = Verdict(since=self.clock())

    def current(self) -> Verdict:
        with self._lock:
            return self._v

    def _set(self, kind, defects=(), reason="", frames=0, latched=False):
        v = self._v
        if (v.kind, v.defects, v.reason, v.latched) != (kind, tuple(defects), reason, latched):
            self._v = Verdict(kind, tuple(defects), reason, frames, self.clock(), latched)
        else:
            self._v = Verdict(kind, tuple(defects), reason, frames, v.since, latched)

    def update(self, valid: bool, defects=(), present: bool | None = None, fault: str = "") -> Verdict:
        """valid: this frame was scored by a healthy model (never True for a failed / stale score).
        present: a bottle is in view (None = unknown, no detector). fault: why the frame is invalid."""
        c = self.cfg
        with self._lock:
            if not valid:
                self._invalid += 1
                if self._invalid >= c.fault_frames:
                    self._frames.clear()
                    self._roll.clear()
                    self._seen = 0
                    self._set(FAULT, reason=fault or "no valid score")
                return self._v
            self._invalid = 0
            d = frozenset(defects)
            if present is None:                                   # no presence signal: rolling vote, no latch
                self._roll.append(d)
                if len(self._roll) < min(c.min_frames, c.window):
                    self._set(CHECKING, frames=len(self._roll))
                else:
                    names = self._vote(list(self._roll))
                    self._set(DEFECT if names else GOOD, names, frames=len(self._roll))
                return self._v
            if not present:
                self._absent += 1
                if self._absent >= c.absent_frames:               # the bottle has left: unlatch
                    self._frames.clear()
                    self._seen = 0
                    self._set(EMPTY)
                return self._v
            self._absent = 0
            self._seen += 1
            if self._v.latched:
                return self._v                                    # one decision per bottle
            if self._v.kind in (EMPTY, FAULT) and not self._frames:
                self._set(CHECKING)
            if self._seen <= c.settle_frames:
                return self._v
            self._frames.append(d)
            if len(self._frames) < c.min_frames:
                self._set(CHECKING, frames=len(self._frames))
                return self._v
            names = self._vote(self._frames)
            self._set(DEFECT if names else GOOD, names, frames=len(self._frames), latched=True)
            return self._v

    def _vote(self, frames: list) -> tuple:
        n = len(frames)
        allnames = sorted({x for f in frames for x in f})
        return tuple(x for x in allnames if sum(x in f for f in frames) / n >= self.cfg.vote)


def combine(verdicts: list) -> Verdict:
    """Several cameras watching one bottle: FAULT > DEFECT > CHECKING > GOOD > NO BOTTLE (a defect only the side
    camera sees is still a defect; any camera that cannot vouch makes the answer FAULT; all must say GOOD)."""
    if not verdicts:
        return Verdict(FAULT, reason="no camera")
    for k in (FAULT, DEFECT):
        hit = [v for v in verdicts if v.kind == k]
        if hit:
            names = tuple(sorted({d for v in hit for d in v.defects}))
            return Verdict(k, names, "; ".join(v.reason for v in hit if v.reason), min(v.frames for v in hit),
                           min(v.since for v in hit), all(v.latched for v in hit))
    kinds = {v.kind for v in verdicts}
    k = CHECKING if CHECKING in kinds else (GOOD if kinds == {GOOD} or (GOOD in kinds and EMPTY not in kinds) else
                                            GOOD if GOOD in kinds else EMPTY)
    return Verdict(k, (), "", min(v.frames for v in verdicts), max(v.since for v in verdicts),
                   all(v.latched for v in verdicts if v.kind in (GOOD, DEFECT)))


def demo():
    t = [0.0]
    T = lambda **k: VerdictTracker(clock=lambda: t[0], **k)                   # noqa: E731
    tr = T()
    assert tr.current().kind == EMPTY
    # empty belt stays NO BOTTLE; classifier noise on an empty belt is ignored
    for _ in range(20):
        assert tr.update(True, ["damaged_bottle"], present=False).kind == EMPTY
    # a good bottle arrives; ONE noisy frame must not flash DEFECT; GOOD is reached and latched
    seq = [set()] * 3 + [{"damaged_bottle"}] + [set()] * 10
    kinds = [tr.update(True, d, present=True).kind for d in seq]
    assert kinds[0] == CHECKING and GOOD in kinds and DEFECT not in kinds, kinds
    assert kinds[-1] == GOOD and tr.current().latched
    # latched: a later burst of defect frames while the SAME bottle is there changes nothing (no flicker)
    for _ in range(10):
        assert tr.update(True, ["tilt_cap"], present=True).kind == GOOD
    # bottle leaves (absent_frames), verdict clears
    for _ in range(7):
        assert tr.update(True, [], present=False).kind == GOOD
    assert tr.update(True, [], present=False).kind == EMPTY
    # a defective bottle: the defect in >= 60 % of frames -> DEFECT with its name, only the consistent defect
    seq = [{"missing_cap"}] * 7 + [{"missing_cap", "water_level"}]
    for d in seq:
        v = tr.update(True, d, present=True)
    assert v.kind == DEFECT and v.defects == ("missing_cap",) and v.text == "DEFECT: Missing cap", (v, v.text)
    # not decided before min_frames
    tr2 = T()
    for _ in range(5):
        assert tr2.update(True, [], present=True).kind == CHECKING
    # failures: FAULT after fault_frames invalid frames, never GOOD; recovers when frames are valid again
    tr3 = T()
    for _ in range(3):
        assert tr3.update(False, fault="camera stopped").kind != FAULT
    assert tr3.update(False, fault="camera stopped").kind == FAULT and "camera stopped" in tr3.current().text
    for _ in range(3):
        tr3.update(True, [], present=True)
    assert tr3.current().kind in (CHECKING, GOOD)
    # no detector (present None): rolling window, flips only on a majority, never latches
    tr4 = T()
    for _ in range(8):
        v = tr4.update(True, [], present=None)
    assert v.kind == GOOD and not v.latched
    for _ in range(10):
        v = tr4.update(True, ["skewed_label"], present=None)
    assert v.kind == DEFECT and v.defects == ("skewed_label",)
    # two cameras
    g, dfx, ck, e = Verdict(GOOD), Verdict(DEFECT, ("tilt_cap",)), Verdict(CHECKING), Verdict(EMPTY)
    assert combine([g, g]).kind == GOOD and combine([g, dfx]).kind == DEFECT and combine([g, ck]).kind == CHECKING
    assert combine([e, e]).kind == EMPTY and combine([g, Verdict(FAULT, reason="x"), dfx]).kind == FAULT
    assert combine([]).kind == FAULT
    # words
    assert pretty("missing_cap") == "Missing cap" and pretty("new_thing") == "New thing"
    assert headline(DEFECT, ["missing_cap", "tilt_cap", "water_level", "skewed_label"]) == \
        "DEFECT: Missing cap, Tilted cap, Wrong water level +1"
    assert shown_result("PASS") == "GOOD" and shown_result("REJECT") == "DEFECT" and shown_result(None) == "--"
    print("ok  verdict: one stable GOOD / DEFECT per bottle (settle, vote, latch until the bottle leaves), "
          "noise on an empty belt ignored, FAULT never GOOD, rolling mode without a detector, camera combine, plain words")


if __name__ == "__main__":
    demo()
