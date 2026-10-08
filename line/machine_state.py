"""THE machine state: one function, one answer, every screen shows the same thing.

    OFFLINE              no PLC link (never connected / disconnected by the operator)
    NOT_READY            something a production start needs is missing (see the readiness list)
    READY                every start check passes; the operator may press START INSPECTION
    INITIALIZING         start pressed: loading models, opening line cameras, waiting for first frames
    RUNNING              the line is answering PLC triggers
    INSPECTING           RUNNING, and at least one bottle is being inspected right now
    STOPPING             stop pressed, the line is winding down
    FAULT                the line is latched (halted / crashed): no PLC command until RESET FAULT
    E_STOP               the hardware E-stop input reads pressed (or the halt came from it)
    COMMUNICATION_FAULT  the line is running (or starting) and the PLC link is lost

The GUI gathers plain facts into a dict (cached state only, no device I/O) and calls state(); widgets
never derive the machine state themselves. readiness() is the start checklist: each item says what is
wrong and what to do. Pure functions.     python machine_state.py   # self-test
"""
from __future__ import annotations

from dataclasses import dataclass

OFFLINE, NOT_READY, READY, INITIALIZING = "OFFLINE", "NOT_READY", "READY", "INITIALIZING"
RUNNING, INSPECTING, STOPPING, FAULT = "RUNNING", "INSPECTING", "STOPPING", "FAULT"
E_STOP, COMMUNICATION_FAULT = "E_STOP", "COMMUNICATION_FAULT"
STATES = (OFFLINE, NOT_READY, READY, INITIALIZING, RUNNING, INSPECTING, STOPPING, FAULT, E_STOP, COMMUNICATION_FAULT)

# what each state means to the operator, and its colour class (theme: PASS / REJECT / FAULT / OFF / ACC)
LOOK = {OFFLINE: ("OFF", "PLC not connected"), NOT_READY: ("WARN", "Not ready to start"),
        READY: ("ACC", "Ready - press START INSPECTION"), INITIALIZING: ("WARN", "Starting..."),
        RUNNING: ("PASS", "Running - waiting for bottles"), INSPECTING: ("PASS", "Inspecting"),
        STOPPING: ("WARN", "Stopping..."), FAULT: ("BAD", "FAULT - see alarms, then RESET FAULT"),
        E_STOP: ("BAD", "EMERGENCY STOP"), COMMUNICATION_FAULT: ("BAD", "PLC communication lost")}

# PLC link states as plc.service reports them
LINK_CONNECTED, LINK_DEGRADED, LINK_FAULT, LINK_DISCONNECTED = "CONNECTED", "DEGRADED", "FAULT", "DISCONNECTED"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    action: str = ""


def readiness(f: dict) -> list:
    """The start checklist from facts f:
        plc_link        CONNECTED / DEGRADED / FAULT / DISCONNECTED
        plc_run         True / False / None (unknown)
        plc_real        True = physical PLC, False = simulator
        cameras         [(name, ok, detail)]  -- chosen line cameras (configured / found)
        models          [(stage, ok, detail)]
        timing          [problem, ...]
        estop           True pressed / False released / None not configured
        recipe          [problem, ...]
    """
    out = []
    link = f.get("plc_link")
    where = "physical PLC" if f.get("plc_real") else "SIMULATOR"
    out.append(Check("PLC connection", link == LINK_CONNECTED,
                     f"{where}: {link or 'not connected'}", "Machine / PLC: Connect"))
    run = f.get("plc_run")
    out.append(Check("PLC in RUN", run is True, {True: "RUN", False: "STOP"}.get(run, "unknown (no PLC data)"),
                     "Switch the PLC to RUN"))
    cams = f.get("cameras") or []
    if not cams:
        out.append(Check("Cameras", False, "no line camera chosen", "Choose Camera 1 / Camera 2"))
    for name, ok, detail in cams:
        out.append(Check(f"Camera: {name}", ok, detail, "Check the camera / Scan cameras"))
    models = f.get("models") or []
    if not models:
        out.append(Check("Model", False, "no inspection task selected", "Choose the AI task"))
    for stage, ok, detail in models:
        out.append(Check(f"Model: {stage}", ok, detail, "Engineer: activate a model"))
    if f.get("presence") is False:
        out.append(Check("Bottle presence check", False,
                         "the classifier alone cannot tell an empty belt from a bottle: it would judge nothing as a bottle",
                         "Choose the AI task 'Classification + Detection'"))
    timing = f.get("timing") or []
    out.append(Check("Timing", not timing, "; ".join(timing) if timing else "valid",
                     "Engineer: Timing / calibration"))
    est = f.get("estop")
    out.append(Check("Emergency stop", est is not True,
                     {True: "PRESSED", False: "released"}.get(est, "no E-stop input configured (hardware only)"),
                     "Release the E-stop"))
    rec = f.get("recipe") or []
    out.append(Check("Recipe", not rec, "; ".join(rec) if rec else "valid", "Engineer: correct the recipe"))
    return out


def state(f: dict) -> tuple:
    """(STATE, reason). Extra facts used here:
        line_running, line_starting, line_stopping, line_halted (reason or None), line_error,
        inspecting (bottles collecting/inspecting now), halted_by_estop (bool)"""
    link = f.get("plc_link")
    running = bool(f.get("line_running"))
    starting = bool(f.get("line_starting"))
    if f.get("estop") is True or (f.get("line_halted") and f.get("halted_by_estop")):
        return E_STOP, f.get("line_halted") or "hardware E-stop input pressed"
    if (running or starting) and link in (LINK_FAULT, LINK_DISCONNECTED):
        return COMMUNICATION_FAULT, f"PLC link {link} while the line is {'running' if running else 'starting'}"
    if f.get("line_error"):
        return FAULT, f["line_error"]
    if running and f.get("line_halted"):
        return FAULT, f["line_halted"]
    if f.get("line_stopping"):
        return STOPPING, "finishing bottles, stopping cameras"
    if starting:
        return INITIALIZING, f.get("starting_detail") or "loading models / opening cameras"
    if running:
        n = int(f.get("inspecting") or 0)
        return (INSPECTING, f"{n} bottle(s) being inspected") if n else (RUNNING, "waiting for the next bottle")
    if link in (None, LINK_DISCONNECTED):
        return OFFLINE, "PLC not connected"
    bad = [c for c in readiness(f) if not c.ok]
    if bad:
        return NOT_READY, "; ".join(f"{c.name}: {c.detail}" for c in bad[:3])
    return READY, "all start checks pass"


def can_start(f: dict) -> bool:
    return state(f)[0] == READY


def demo():
    ok = {"plc_link": LINK_CONNECTED, "plc_run": True, "plc_real": False,
          "cameras": [("Camera 1", True, "found")], "models": [("detection", True, "stage2_best.pt")],
          "timing": [], "estop": False, "recipe": []}
    assert state(ok) == (READY, "all start checks pass") and can_start(ok)
    assert all(c.ok for c in readiness(ok))
    assert state({})[0] == OFFLINE
    s, why = state(dict(ok, plc_run=False))
    assert s == NOT_READY and "PLC in RUN" in why, why
    assert state(dict(ok, timing=["T0 >= travel"]))[0] == NOT_READY
    assert state(dict(ok, cameras=[]))[0] == NOT_READY
    assert state(dict(ok, models=[("detection", False, "weights missing")]))[0] == NOT_READY
    assert state(dict(ok, estop=True))[0] == E_STOP
    assert state(dict(ok, estop=None))[0] == READY                      # no E-stop input: hardware-only, allowed
    assert state(dict(ok, line_starting=True))[0] == INITIALIZING
    assert state(dict(ok, line_running=True))[0] == RUNNING
    assert state(dict(ok, line_running=True, inspecting=1))[0] == INSPECTING
    assert state(dict(ok, line_running=True, line_halted="operator STOP"))[0] == FAULT
    assert state(dict(ok, line_running=True, line_halted="E-stop X3", halted_by_estop=True))[0] == E_STOP
    assert state(dict(ok, line_running=True, plc_link=LINK_FAULT))[0] == COMMUNICATION_FAULT
    assert state(dict(ok, line_stopping=True))[0] == STOPPING
    assert state(dict(ok, line_error="machine cycle crashed"))[0] == FAULT
    assert set(LOOK) == set(STATES)
    print("ok  machine state: one state from one set of facts (10 states), start checklist, precedence "
          "E_STOP > COMMUNICATION_FAULT > FAULT > STOPPING > INITIALIZING > RUNNING/INSPECTING > OFFLINE/NOT_READY/READY")


if __name__ == "__main__":
    demo()
