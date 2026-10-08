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
# UNSURE: a defect score sat just below its threshold in most frames. Neither "GOOD" nor a named defect the model is
# not sure of: the bottle is shown as CHECK and should be looked at (and rejected on a line: never a silent pass).
UNSURE = "CHECK"
UNSURE_MARK = "?"                    # a frame's defect list carries "?missing_cap" for "close to the threshold"

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
    if kind == UNSURE:
        names = [pretty(d) + "?" for d in defects]
        return "CHECK: " + (", ".join(names[:3]) if names else "unsure")
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
    bottle: int = 0                  # counts latched bottles (one per bottle): lets a consumer act once per bottle

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
            self._bottle = 0
            self._v = Verdict(since=self.clock())

    def current(self) -> Verdict:
        with self._lock:
            return self._v

    def _set(self, kind, defects=(), reason="", frames=0, latched=False):
        v = self._v
        if latched and not v.latched:
            self._bottle += 1                                     # a new bottle has been decided
        if (v.kind, v.defects, v.reason, v.latched) != (kind, tuple(defects), reason, latched):
            self._v = Verdict(kind, tuple(defects), reason, frames, self.clock(), latched, self._bottle)
        else:
            self._v = Verdict(kind, tuple(defects), reason, frames, v.since, latched, self._bottle)

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
                    kind, names = self._decide(list(self._roll))
                    self._set(kind, names, frames=len(self._roll))
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
            kind, names = self._decide(self._frames)
            self._set(kind, names, frames=len(self._frames), latched=True)
            return self._v

    def _decide(self, frames: list) -> tuple:
        """(kind, names): DEFECT if a defect is in >= vote of the frames; else CHECK if a defect was at or near its
        threshold (sure or unsure) in >= vote of them; else GOOD."""
        n = len(frames)
        sure = [{x for x in f if not x.startswith(UNSURE_MARK)} for f in frames]
        near = [{x.lstrip(UNSURE_MARK) for x in f} for f in frames]
        names = self._vote(sure, n)
        if names:
            return DEFECT, names
        unsure = self._vote(near, n)
        return (UNSURE, unsure) if unsure else (GOOD, ())

    def _vote(self, frames: list, n: int) -> tuple:
        allnames = sorted({x for f in frames for x in f})
        return tuple(x for x in allnames if sum(x in f for f in frames) / n >= self.cfg.vote)


def combine(verdicts: list) -> Verdict:
    """Several cameras watching one bottle: FAULT > DEFECT > CHECKING > GOOD > NO BOTTLE (a defect only the side
    camera sees is still a defect; any camera that cannot vouch makes the answer FAULT; all must say GOOD)."""
    if not verdicts:
        return Verdict(FAULT, reason="no camera")
    for k in (FAULT, DEFECT, UNSURE):
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
    # one bottle number per latched bottle (Auto-collect saves exactly one frame per bottle)
    tr5 = T()
    for _ in range(10):
        v1 = tr5.update(True, [], present=True)
    assert v1.latched and v1.bottle == 1
    for _ in range(8):
        tr5.update(True, [], present=False)
    for _ in range(10):
        v2 = tr5.update(True, [], present=True)
    assert v2.bottle == 2 and tr5.update(True, [], present=True).bottle == 2
    # unsure: a score just under the threshold in most frames -> CHECK, never GOOD and never a named DEFECT
    tr6 = T()
    for _ in range(10):
        v = tr6.update(True, ["?damaged_bottle"], present=True)
    assert v.kind == UNSURE and v.defects == ("damaged_bottle",) and v.text == "CHECK: Damaged bottle?", (v, v.text)
    tr7 = T()                                       # sure in some frames, unsure in the rest: still not a confident DEFECT
    for i in range(10):
        v = tr7.update(True, ["missing_cap"] if i % 3 == 0 else ["?missing_cap"], present=True)
    assert v.kind == UNSURE, v
    assert combine([Verdict(GOOD), Verdict(UNSURE, ("tilt_cap",))]).kind == UNSURE
    assert combine([Verdict(UNSURE, ("tilt_cap",)), Verdict(DEFECT, ("missing_cap",))]).kind == DEFECT
    # words
    assert pretty("missing_cap") == "Missing cap" and pretty("new_thing") == "New thing"
    assert headline(DEFECT, ["missing_cap", "tilt_cap", "water_level", "skewed_label"]) == \
        "DEFECT: Missing cap, Tilted cap, Wrong water level +1"
    assert shown_result("PASS") == "GOOD" and shown_result("REJECT") == "DEFECT" and shown_result(None) == "--"
    print("ok  verdict: one stable GOOD / DEFECT per bottle (settle, vote, latch until the bottle leaves), "
          "noise on an empty belt ignored, FAULT never GOOD, CHECK for unsure bottles, one number per bottle, "
          "rolling mode without a detector, camera combine, plain words")


if __name__ == "__main__":
    demo()
