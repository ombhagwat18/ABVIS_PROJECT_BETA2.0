"""Machine alarms: one catalogue of codes, one manager, one history.

Every alarm has a code, a severity, the time it was raised, the cause (the technical detail), an
operator message and a recommended action, and is either ACTIVE, ACKNOWLEDGED (still true, operator
has seen it) or CLEARED. Two kinds:

  * condition alarms (PLC_DISCONNECTED, CAMERA_STALE, ...) are raised while a condition holds and
    cleared by the code that watches it (set_condition);
  * event alarms (REJECT_DEADLINE_MISSED, BOTTLE_UNTRIGGERED, ...) happened once; they stay active
    until the operator acknowledges them (RESET FAULT acknowledges all).

The same code + key raised again while active is counted, not duplicated (no alarm floods at the
camera frame rate). Persistence is a callback (production_store.ProductionStore.log_alarm), so this
module has no I/O.     python alarms.py   # self-test
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

CRITICAL, MAJOR, WARNING, INFO = "CRITICAL", "MAJOR", "WARNING", "INFO"
SEVERITY_RANK = {CRITICAL: 0, MAJOR: 1, WARNING: 2, INFO: 3}

# code -> (severity, operator message, recommended action)
CATALOG = {
    "E_STOP": (CRITICAL, "Emergency stop active", "Make the machine safe, release the E-stop, then RESET FAULT."),
    "LINE_HALTED": (CRITICAL, "Inspection line halted", "Find the cause in the alarm list, fix it, then RESET FAULT."),
    "TOO_MANY_FAULTS": (CRITICAL, "Too many consecutive FAULT bottles",
                        "Check cameras, lighting and model; remove the FAULT bottles; then RESET FAULT."),
    "MACHINE_CYCLE_CRASHED": (CRITICAL, "Inspection software stopped unexpectedly",
                              "Stop the line, save the log, restart the application."),
    "PLC_DISCONNECTED": (MAJOR, "PLC not connected", "Check the PLC cable / COM port and press Connect."),
    "PLC_COMM_FAULT": (CRITICAL, "PLC communication lost", "Check the cable and that no other program holds the port."),
    "PLC_NOT_RUNNING": (MAJOR, "PLC is in STOP", "Put the PLC in RUN (RUN/STOP switch or ISPSoft)."),
    "PLC_ACK_TIMEOUT": (MAJOR, "PLC did not confirm a command", "Remove the bottle by hand; check the PLC program and link."),
    "PLC_COMMAND_FAILED": (MAJOR, "PLC command could not be sent", "Remove the bottle by hand; check the PLC link."),
    "CAMERA_DISCONNECTED": (MAJOR, "Camera not delivering images", "Check the camera cable / USB port, then restart the line."),
    "CAMERA_STALE": (MAJOR, "Camera image is old", "Check the camera; another program may be using it."),
    "CAMERA_BANDWIDTH_PROBLEM": (MAJOR, "Two cameras cannot stream together",
                                 "Put each camera on its own USB root port (USB 3), or lower the resolution."),
    "CAMERA_ASSOCIATION_FAULT": (MAJOR, "Camera images could not be matched to the bottle",
                                 "Increase the gap between bottles or re-calibrate the conveyor speed."),
    "MODEL_NOT_FOUND": (MAJOR, "Inspection model missing", "Activate a model (engineer: Models / Train)."),
    "MODEL_RUNTIME_ERROR": (MAJOR, "Inspection model failed", "Stop the line and check the GPU / model files."),
    "NO_BOTTLE": (WARNING, "No bottle seen after a trigger", "Check the sensor position and the camera view."),
    "TRIGGER_MISSED": (WARNING, "Inspection trigger lost", "Check the PLC link; remove the bottle by hand."),
    "BOTTLE_UNTRIGGERED": (MAJOR, "Bottle passed without inspection",
                           "Remove the bottle by hand. Bottles are too close (one-bottle PLC handshake)."),
    "TIMING_INVALID": (MAJOR, "Line timing is not valid", "Engineer: calibrate speed / distances (Timing)."),
    "REJECT_DEADLINE_MISSED": (MAJOR, "Reject too late: bottle NOT ejected", "Remove the bottle by hand."),
    "REJECT_NOT_OBSERVED": (MAJOR, "Reject output not seen", "Check the reject cylinder and the PLC program."),
    "UNEXPECTED_REJECT": (WARNING, "Reject output fired with no reject pending", "Check the PLC program / wiring."),
    "RECIPE_INVALID": (MAJOR, "Inspection recipe is not valid", "Engineer: correct the recipe."),
    "LOG_WRITE_FAILED": (WARNING, "Production record not written", "Check free disk space."),
    "DISK_LOW": (WARNING, "Disk space low", "Free disk space; evidence images stop when the disk is full."),
}

ACTIVE, ACKNOWLEDGED, CLEARED = "ACTIVE", "ACKNOWLEDGED", "CLEARED"


@dataclass
class Alarm:
    code: str
    key: str
    severity: str
    message: str
    action: str
    cause: str
    raised: float                      # epoch seconds
    state: str = ACTIVE
    count: int = 1
    last: float = 0.0
    acked: float | None = None
    cleared: float | None = None
    source: str = ""
    condition: bool = False
    seq: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.state != CLEARED

    def row(self) -> dict:
        return {"code": self.code, "key": self.key, "severity": self.severity, "state": self.state,
                "message": self.message, "action": self.action, "cause": self.cause, "source": self.source,
                "raised": self.raised, "last": self.last, "count": self.count, "acked": self.acked,
                "cleared": self.cleared}


def describe(code: str) -> tuple:
    return CATALOG.get(code, (MAJOR, code.replace("_", " ").capitalize(), "See the alarm cause."))


class AlarmManager:
    """Thread-safe. on_change(alarm) is called (outside the lock) on raise / ack / clear."""

    def __init__(self, on_change=None, history: int = 1000, clock=time.time):
        self._lock = threading.Lock()
        self._active: dict = {}                 # (code, key) -> Alarm
        self._history: list = []
        self._max = history
        self._seq = 0
        self.on_change = on_change
        self.clock = clock

    def raise_(self, code: str, cause: str = "", key: str = "", source: str = "", condition: bool = False) -> Alarm:
        sev, msg, act = describe(code)
        now = self.clock()
        with self._lock:
            a = self._active.get((code, key))
            if a is not None:
                a.count += 1
                a.last, a.cause = now, cause or a.cause
                if a.state == ACKNOWLEDGED and not condition:
                    a.state = ACTIVE                   # an event that happens again needs a new acknowledgement
                new = False
            else:
                self._seq += 1
                a = Alarm(code, key, sev, msg, act, cause, now, last=now, source=source, condition=condition,
                          seq=self._seq)
                self._active[(code, key)] = a
                self._history.append(a)
                del self._history[:-self._max]
                new = True
        if new:
            self._notify(a)
        return a

    def set_condition(self, code: str, on: bool, cause: str = "", key: str = "", source: str = ""):
        """Raise while `on`, clear when it stops being true. Cheap to call every UI tick."""
        if on:
            with self._lock:
                a = self._active.get((code, key))
            if a is None:
                self.raise_(code, cause, key, source, condition=True)
            elif cause and a.cause != cause:
                a.cause = cause
        else:
            self.clear(code, key)

    def clear(self, code: str, key: str = "") -> bool:
        with self._lock:
            a = self._active.pop((code, key), None)
            if a is None:
                return False
            a.state, a.cleared = CLEARED, self.clock()
        self._notify(a)
        return True

    def acknowledge_all(self) -> int:
        """Operator acknowledgement: event alarms are cleared (they are over), condition alarms that
        still hold stay visible as ACKNOWLEDGED. Returns how many changed."""
        now = self.clock()
        changed = []
        with self._lock:
            for k, a in list(self._active.items()):
                if a.condition:
                    if a.state == ACTIVE:
                        a.state, a.acked = ACKNOWLEDGED, now
                        changed.append(a)
                else:
                    a.state, a.acked, a.cleared = CLEARED, now, now
                    del self._active[k]
                    changed.append(a)
        for a in changed:
            self._notify(a)
        return len(changed)

    def active(self) -> list:
        """Active alarms, most severe then newest first."""
        with self._lock:
            out = list(self._active.values())
        return sorted(out, key=lambda a: (SEVERITY_RANK.get(a.severity, 9), -a.last))

    def history(self, n: int = 200) -> list:
        with self._lock:
            return list(self._history[-n:])

    def worst(self) -> str | None:
        act = [a for a in self.active() if a.state == ACTIVE]
        return act[0].severity if act else None

    def _notify(self, a: Alarm):
        if self.on_change:
            try:
                self.on_change(a)
            except Exception:                          # noqa: BLE001 - persistence must never break alarming
                pass


def demo():
    t = [1000.0]
    seen = []
    m = AlarmManager(on_change=lambda a: seen.append((a.code, a.state)), clock=lambda: t[0])
    a = m.raise_("REJECT_DEADLINE_MISSED", "bottle 000007 late by 420 ms", key="000007")
    assert a.severity == MAJOR and "NOT ejected" in a.message and a.action
    m.raise_("REJECT_DEADLINE_MISSED", "again", key="000007")
    assert len(m.active()) == 1 and m.active()[0].count == 2              # counted, not duplicated
    m.set_condition("PLC_DISCONNECTED", True, "COM5: access denied")
    m.set_condition("PLC_DISCONNECTED", True, "COM5: access denied")
    assert len(m.active()) == 2 and {x.code for x in m.active()} == {"REJECT_DEADLINE_MISSED", "PLC_DISCONNECTED"}
    m.raise_("E_STOP", "X3 open")
    assert m.active()[0].code == "E_STOP" and m.worst() == CRITICAL       # most severe first
    m.clear("E_STOP")
    assert m.acknowledge_all() == 2                                      # event cleared, condition acknowledged
    act = m.active()
    assert [x.code for x in act] == ["PLC_DISCONNECTED"] and act[0].state == ACKNOWLEDGED, act
    assert m.worst() is None                                             # acknowledged: nothing unseen
    m.set_condition("PLC_DISCONNECTED", False)
    assert not m.active() and len(m.history()) == 3
    assert ("PLC_DISCONNECTED", CLEARED) in seen and ("E_STOP", ACTIVE) in seen
    assert describe("SOMETHING_NEW")[1] == "Something new"               # unknown code still readable
    for code, (sev, msg, act_) in CATALOG.items():
        assert sev in SEVERITY_RANK and msg and act_, code
    print(f"ok  alarms: {len(CATALOG)} coded alarms (severity, message, action), dedup, condition vs event, "
          f"acknowledge, clear, history")


if __name__ == "__main__":
    demo()
