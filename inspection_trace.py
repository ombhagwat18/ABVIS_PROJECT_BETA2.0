"""Traceability: one InspectionRecord per inspection event, and a bounded in-process store.

This layer RECORDS what infer.Camera decided; it never decides PASS/REJECT/FAULT
itself. The states are infer's own constants, imported, not redefined.

    Camera.inspection() -> InspectionRecord.from_inspection() -> TraceStore.record()

Nothing here touches disk, a database, a PLC, or images. A field the system cannot
supply is None / empty -- never a guess.

    python inspection_trace.py   # self-test
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, fields

import dataset as D
import infer
from infer import FAULT, PASS, REJECT           # the one definition of the safety states

STATES = (PASS, REJECT, FAULT)

# Identifies this process launch. Camera sessions and sequence numbers restart from
# 1 every time the app starts, so without it inspection ids from two launches would
# collide as soon as records are persisted.
RUN_ID = uuid.uuid4().hex[:8]

_fault_n = itertools.count(1)                    # next() is atomic under the GIL
_TUPLES = ("reasons", "hits")


def make_inspection_id(camera_id: str, session: int | None, seq: int | None,
                       run_id: str = RUN_ID) -> str:
    """<run>-<camera>-<session>-<seq> for an inspected frame.

    Unique across camera restarts (session differs) and across app launches (run
    differs). With no inspected frame to number there is nothing to key on, so a
    process-wide counter takes its place: `...-F<n>`.
    """
    tail = str(seq) if seq is not None else f"F{next(_fault_n)}"
    return f"{run_id}-{camera_id}-{session if session is not None else 0}-{tail}"


@dataclass(frozen=True)
class InspectionRecord:
    inspection_id: str
    timestamp: float                    # wall clock, epoch seconds: for humans/reports only,
    #                                     never for freshness or latency
    state: str                          # PASS | REJECT | FAULT -- what the camera concluded
    decision: str                       # PASS | REJECT | FAULT -- what is to be done; today
    #                                     identical to state (no decision policy exists yet)
    camera_id: str
    session_id: int | None              # Camera.session: counts start()s of that camera
    frame_seq: int | None               # newest frame captured when this was taken
    frame_ts: float | None              # ...its time.monotonic() stamp
    result_seq: int | None              # the frame that was actually scored
    result_ts: float | None
    reasons: tuple = ()                 # why FAULT (empty otherwise)
    hits: tuple = ()                    # defects at/above threshold (REJECT)
    run_id: str = RUN_ID
    project_id: str | None = None
    job_id: str | None = None           # no job system yet: always None for now
    model_id: str | None = None         # checkpoint stamp
    processing_ms: float | None = None  # inference time of the scored frame (perf_counter);
    #                                     NOT capture-to-result latency
    evidence_path: str | None = None    # reserved: nothing saves images yet

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {self.state!r}")
        if self.decision not in STATES:
            raise ValueError(f"decision must be one of {STATES}, got {self.decision!r}")
        if not self.inspection_id:
            raise ValueError("inspection_id is required")
        for name in _TUPLES:                     # frozen, so no caller can mutate a stored record
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_inspection(cls, insp: infer.Inspection, *, decision: str | None = None,
                        job_id: str | None = None, project_id: str | None = None,
                        evidence_path: str | None = None, timestamp: float | None = None,
                        run_id: str = RUN_ID) -> "InspectionRecord":
        """Build from Camera.inspection(). decision defaults to the state; project_id to
        the active project. Only PASS/REJECT name an inspected frame, so only those get
        a deterministic id (and so can be recognised if the same result is seen twice)."""
        scored = insp.state in (PASS, REJECT)
        return cls(
            inspection_id=make_inspection_id(insp.camera_id, insp.session,
                                             insp.result_seq if scored else None, run_id),
            timestamp=time.time() if timestamp is None else timestamp,
            state=insp.state,
            decision=insp.state if decision is None else decision,
            camera_id=insp.camera_id, session_id=insp.session,
            frame_seq=insp.frame_seq, frame_ts=insp.frame_ts,
            result_seq=insp.result_seq, result_ts=insp.result_ts,
            reasons=(insp.reason,) if insp.reason else (),
            hits=tuple(insp.hits), run_id=run_id,
            project_id=(D.PROJECT or None) if project_id is None else project_id,
            job_id=job_id, model_id=insp.model_id, processing_ms=insp.infer_ms,
            evidence_path=evidence_path)

    def to_dict(self) -> dict:
        """Plain JSON-able dict (tuples become lists)."""
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "InspectionRecord":
        """Inverse of to_dict. Unknown keys are ignored; a missing required key is an error."""
        known = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in d.items() if k in known})
        except TypeError as e:
            raise ValueError(f"cannot build an InspectionRecord: {e}") from e


class TraceStore:
    """Bounded, thread-safe, in-memory history. Oldest records fall off the far end.

    It stores what it is given: it does not deduplicate, so a caller polling
    Camera.inspection() must record each new result_seq once.
    """

    def __init__(self, max_history: int = 1000):
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_history = max_history
        self._items: deque[InspectionRecord] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._evicted = 0

    def record(self, rec: InspectionRecord) -> InspectionRecord:
        if not isinstance(rec, InspectionRecord):
            raise TypeError(f"expected InspectionRecord, got {type(rec).__name__}")
        with self._lock:
            if len(self._items) == self.max_history:
                self._evicted += 1
            self._items.append(rec)
        return rec

    def latest(self) -> InspectionRecord | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def recent(self, limit: int = 50) -> list[InspectionRecord]:
        """Newest first."""
        if limit <= 0:
            return []
        with self._lock:
            return list(itertools.islice(reversed(self._items), limit))

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def evicted(self) -> int:
        """Records pushed out by the bound since the last clear(): history is incomplete if > 0."""
        with self._lock:
            return self._evicted

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._evicted = 0


# ---------------------------------------------------------------------------- self-test

def _insp(state=PASS, *, camera="0", session=1, fseq=9, rseq=8, hits=(), reason=None,
          model="m1", ms=12.5):
    scored = state in (PASS, REJECT)
    return infer.Inspection(camera, session, state, list(hits), reason, fseq, 100.5,
                            rseq if scored else None, 100.4 if scored else None,
                            model, ms if scored else None)


def demo():
    # 1-3. PASS / REJECT / FAULT, each preserving the metadata it was given
    p = InspectionRecord.from_inspection(_insp(PASS), job_id=None)
    assert (p.state, p.decision, p.hits, p.reasons) == (PASS, PASS, (), ())
    assert (p.camera_id, p.session_id, p.frame_seq, p.frame_ts) == ("0", 1, 9, 100.5)
    assert (p.result_seq, p.result_ts, p.model_id, p.processing_ms) == (8, 100.4, "m1", 12.5)
    assert p.job_id is None and p.evidence_path is None and p.run_id == RUN_ID
    assert p.project_id == (D.PROJECT or None) and p.timestamp > 1.6e9
    r = InspectionRecord.from_inspection(_insp(REJECT, hits=["tilt_cap", "water_level"]))
    assert (r.state, r.decision, r.hits, r.reasons) == (REJECT, REJECT, ("tilt_cap", "water_level"), ())
    f = InspectionRecord.from_inspection(_insp(FAULT, reason="stale score (>1s old)"))
    assert (f.state, f.decision, f.reasons, f.hits) == (FAULT, FAULT, ("stale score (>1s old)",), ())
    assert f.result_seq is None and f.processing_ms is None, "a FAULT must not invent a score/timing"
    assert f.frame_seq == 9, "...but keeps the frame metadata it does know"
    for bad in ("MAYBE", None):
        try:
            InspectionRecord.from_inspection(_insp(PASS)).__class__(
                **{**p.to_dict(), "state": bad})
            raise SystemExit(f"state {bad!r} was accepted")
        except ValueError:
            pass

    # 4-5. ids: unique per frame, and never colliding across camera restarts / launches
    a = InspectionRecord.from_inspection(_insp(PASS, rseq=8))
    assert a.inspection_id == p.inspection_id, "same inspected frame -> same id (deterministic)"
    ids = {a.inspection_id,
           InspectionRecord.from_inspection(_insp(PASS, rseq=9)).inspection_id,         # next frame
           InspectionRecord.from_inspection(_insp(PASS, session=2)).inspection_id,      # restart, same seq
           InspectionRecord.from_inspection(_insp(PASS, camera="1")).inspection_id,     # other camera
           InspectionRecord.from_inspection(_insp(PASS), run_id="deadbeef").inspection_id}  # app relaunch
    assert len(ids) == 5, ids
    f1 = InspectionRecord.from_inspection(_insp(FAULT, reason="x")).inspection_id
    f2 = InspectionRecord.from_inspection(_insp(FAULT, reason="x")).inspection_id
    assert f1 != f2, "two FAULT events must not share an id"

    # 7. serialization round trip, including through real JSON
    for rec in (p, r, f):
        d = rec.to_dict()
        assert isinstance(d["hits"], list) and isinstance(d["reasons"], list)
        assert InspectionRecord.from_dict(d) == rec
        assert InspectionRecord.from_dict(json.loads(json.dumps(d))) == rec
    assert InspectionRecord.from_dict({**p.to_dict(), "future_field": 1}) == p
    try:
        InspectionRecord.from_dict({"state": PASS})
        raise SystemExit("incomplete dict was accepted")
    except ValueError:
        pass

    # 8-9. bounded, ordered store
    st = TraceStore(max_history=5)
    assert st.latest() is None and st.count() == 0 and st.recent() == []
    recs = [InspectionRecord.from_inspection(_insp(PASS, rseq=i)) for i in range(1, 9)]
    for x in recs:
        st.record(x)
    assert st.count() == 5 and st.evicted == 3, (st.count(), st.evicted)
    assert st.latest() is recs[-1]
    assert [x.result_seq for x in st.recent(10)] == [8, 7, 6, 5, 4], "newest first, oldest evicted"
    assert [x.result_seq for x in st.recent(2)] == [8, 7] and st.recent(0) == []
    st.clear()
    assert st.count() == 0 and st.latest() is None and st.evicted == 0
    try:
        st.record({"not": "a record"})
        raise SystemExit("store accepted a non-record")
    except TypeError:
        pass

    # 10. concurrent writers: nothing lost, nothing duplicated, per-writer order kept.
    # A 1 us switch interval makes the interpreter swap threads constantly, so a missing
    # lock shows up as a wrong count instead of hiding behind the default 5 ms.
    import sys
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        _concurrent(recs)
    finally:
        sys.setswitchinterval(old_switch)

    _integration()
    print("ok  (PASS/REJECT/FAULT records, ids across restarts, round-trip, bounded ordered "
          "thread-safe store, live Camera -> record)")


def _concurrent(recs):
    big, n_threads, per = TraceStore(max_history=10_000), 8, 500
    barrier = threading.Barrier(n_threads)

    def writer(t):
        barrier.wait()
        for i in range(per):
            big.record(InspectionRecord.from_inspection(_insp(PASS, camera=f"w{t}", rseq=i)))
    ts = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    got = list(reversed(big.recent(10_000)))
    assert big.count() == n_threads * per == len(got)
    assert len({x.inspection_id for x in got}) == n_threads * per
    for t in range(n_threads):
        seqs = [x.result_seq for x in got if x.camera_id == f"w{t}"]
        assert seqs == list(range(per)), f"writer {t} out of order"
    small = TraceStore(max_history=100)
    barrier = threading.Barrier(n_threads)
    ts = [threading.Thread(target=lambda t=t: (barrier.wait(), [small.record(recs[0]) for _ in range(per)]))
          for t in range(n_threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert small.count() == 100 and small.evicted == n_threads * per - 100, (small.count(), small.evicted)


def _integration():
    """A real Camera thread (fake capture + model) -> Inspection -> record. Proves the
    wiring, that the record agrees with result(), and that a restart cannot collide."""
    import numpy as np
    box = {"infer": "pass"}

    class Cap:
        def __init__(self): self.n = 0
        def isOpened(self): return True
        def get(self, *_): return 0
        def set(self, *_): pass
        def release(self): pass
        def read(self):
            time.sleep(0.004)
            self.n += 1
            return True, np.zeros((8, 8, 3), np.uint8)

    class Model:
        defects, roi, stamp = ["a"], None, "fake-model"
        def predict(self, frame):
            time.sleep(0.003)
            if box["infer"] == "raise":
                raise RuntimeError("boom")
            return {"a": 0.9 if box["infer"] == "reject" else 0.1}

    def wait(cond, what):
        t0 = time.monotonic()
        while not cond():
            assert time.monotonic() - t0 < 3, f"timed out: {what}"
            time.sleep(0.01)

    real_open, real_cfg = infer.open_capture, D.load_config
    infer.open_capture, D.load_config = (lambda src: Cap()), (lambda: {"thresholds": {}})
    cam = infer.Camera("t")
    cam.model = Model()
    store = TraceStore()
    try:
        cam.start(0)
        wait(lambda: cam.result()[0] == PASS, "PASS")
        rec = store.record(InspectionRecord.from_inspection(cam.inspection()))
        state, hits, reason = cam.result()
        assert rec.state == PASS and rec.camera_id == "0" and rec.session_id == cam.session
        assert rec.model_id == "fake-model" and rec.result_seq >= 1 and rec.frame_seq >= rec.result_seq
        assert rec.processing_ms is not None and 0 < rec.processing_ms < 1000, rec.processing_ms
        first_id = rec.inspection_id

        box["infer"] = "reject"
        wait(lambda: cam.result()[0] == REJECT, "REJECT")
        rej = store.record(InspectionRecord.from_inspection(cam.inspection()))
        assert rej.state == REJECT and rej.hits == ("a",) and rej.reasons == ()

        box["infer"] = "raise"
        wait(lambda: cam.result()[0] == FAULT, "FAULT")
        flt = store.record(InspectionRecord.from_inspection(cam.inspection()))
        assert flt.state == FAULT and "inference failed" in flt.reasons[0]
        assert flt.processing_ms is None and flt.result_seq is None and flt.model_id == "fake-model"

        # restart the camera: sequence starts over, ids must not repeat
        box["infer"] = "pass"
        cam.stop(); cam.start(0)
        wait(lambda: cam.result()[0] == PASS, "PASS after restart")
        again = store.record(InspectionRecord.from_inspection(cam.inspection()))
        assert again.session_id == rec.session_id + 1
        assert again.inspection_id != first_id
        ids = [x.inspection_id for x in store.recent(10)]
        assert len(ids) == len(set(ids)) == 4
        assert [x.state for x in store.recent(10)] == [PASS, FAULT, REJECT, PASS], "newest first"
    finally:
        cam.stop()
        infer.open_capture, D.load_config = real_open, real_cfg


if __name__ == "__main__":
    demo()
