"""Timing check: does the line's timing (PLC ladder + settings + cameras + AI) work at the exact moments it must?

    timing_check.bat                                   # double-click: menu
    python -m tools.timing_check                       # 1. PLAN: settings + the ladder in plc_ladder, no hardware at all
    python -m tools.timing_check --apply               #    + write the ladder-derived settings (backup kept)
    python -m tools.timing_check --spacing 600         #    + check a real bottle spacing (mm) against the answer time
    python -m tools.timing_check --watch 120           # 2. WATCH the real PLC for 120 s (READ ONLY): X0/M2/M0/M1/Y0 timeline
    python -m tools.timing_check --travel 8            # 3. STOPWATCH: press Enter when each bottle reaches the cylinder
    python -m tools.timing_check --selftest

The PLAN is arithmetic on what is saved: ladder file (plc/ladder_check decodes it), settings.json, and the measured
constants of this PC (AI time per frame). WATCH and TRAVEL read the PLC through PLCService (no write of any kind: no
M0 / M1, no Y) so the numbers are MEASURED; close NexaInspect, ISPSoft and COMMGR first (one program per COM port).

Time zero is the photo-eye (X0 rising). Everything below is seconds after it:
    M2 seen -> frames watched (capture delay .. + window) -> AI done -> M0 (PASS) or M1 (REJECT) written at `travel`
    -> San_test ladder: Y0 ON at once for T0 -> done. (final_year ladder: M1 is written T0 earlier, Y0 starts T0 later.)
Each line of the result is OK / WARN / FAIL / INFO and says whether the number is SAVED (settings / ladder),
ESTIMATE (arithmetic on constants) or MEASURED (this run). The report is also written to reports/timing/.
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"

# Constants of this installation. EST = estimate (arithmetic), the rest come from measurements made earlier.
AI_FRAME_S = 0.15                 # classifier ~70 ms + YOLO ~80 ms per frame (measured 2026-10-07, RTX 3050)
SERIAL_TXN_S = 0.045              # one Modbus ASCII transaction at 9600 baud 7E1 (~32 chars x 10 bit = 33 ms + PLC turnaround)
WRITE_PATH_S = 0.15               # M1 write = precondition read + write (PLCService re-reads M2/M0/M1 first)
SCAN_S = 0.02                     # DVP-SS2 scan, ladder this size
HEARTBEAT_PERIOD_S = 1.0          # plc.service.HeartbeatPump default
STUCK_EXTRA_S = 3.0               # machine_cycle: REJECT_NOT_OBSERVED after cycle + this
BOTTLE_MM = 65.0                  # 250 ml bottle diameter (ESTIMATE, only used for the "can the pusher still hit it" sentence)


class Row:
    def __init__(self, rid, status, title, detail, source="SAVED"):
        self.rid, self.status, self.title, self.detail, self.source = rid, status, title, detail, source


# ============================================================================================== PLAN
def _ladder(settings: dict):
    from plc import ladder_check as LC
    path = LC.isp_for(settings.get("plc_ladder"))
    nets, _ = LC.load(path)
    t = {}
    for name in ("T0", "T1", "T2"):
        k = LC._timer(nets, name)[1]
        t[name] = None if k is None else round(int(k) * LC.TIMER_BASE_S, 3)
    return path, nets, LC.style(nets), t


def analyse(settings: dict, spacing_mm: float = 0.0) -> tuple:
    """(rows, timeline, wanted settings). Pure: reads files, writes nothing."""
    from line import machine_cycle as MC
    from line import tracking as TR
    rows: list = []
    cfg = MC.line_settings(settings)
    path, nets, kind, lt = _ladder(settings)
    want_kind = MC.ladder_kind(cfg)
    san = kind == "san_test"
    t0, t1 = float(cfg["plc_t0_s"]), float(cfg["plc_t1_s"])

    # ---- the ladder and the settings must describe the same machine
    from plc import ladder_check as LC
    saved = {k: f.stat().st_mtime for k, f in LC.ISP_FILES.items() if f.exists()}
    newest = max(saved, key=saved.get) if saved else want_kind
    stamp = lambda k: time.strftime("%d %b %H:%M", time.localtime(saved[k]))                 # noqa: E731
    ok = kind == want_kind and newest == want_kind
    rows.append(Row("T1", OK if ok else (FAIL if kind != want_kind else WARN), "Ladder design = settings plc_ladder",
                    f"settings plc_ladder = '{want_kind}' -> {path.name} ({kind} design, saved {stamp(want_kind) if want_kind in saved else '?'}); "
                    f"the most recently saved ladder project is '{newest}'"
                    + ("" if ok else ". If the newest one is what is in the PLC, set plc_ladder to it (the REJECT is timed for the "
                       "design named there)."), "SAVED"))
    if san:
        ok = lt["T0"] is not None and abs(lt["T0"] - t0) < 0.05
        rows.append(Row("T2", OK if ok else FAIL, "T0 (Y0 pulse length) in the ladder = settings plc_t0_s",
                        f"ladder T0 = {lt['T0']} s, settings plc_t0_s = {t0:g} s", "SAVED"))
    else:
        ok = lt["T0"] is not None and lt["T1"] is not None and abs(lt["T0"] - t0) < 0.05 and abs(lt["T1"] - t1) < 0.05
        rows.append(Row("T2", OK if ok else FAIL, "T0 / T1 in the ladder = settings plc_t0_s / plc_t1_s",
                        f"ladder T0 {lt['T0']} s T1 {lt['T1']} s; settings {t0:g} / {t1:g} s", "SAVED"))

    # ---- travel (photo-eye -> reject cylinder)
    travel, measured = MC.travel_time(cfg)
    src = ("reject_travel_s (stopwatch)" if float(cfg.get("reject_travel_s") or 0) > 0 else
           f"{cfg['inspection_to_reject_mm']:g} mm / {cfg['conveyor_mm_s']:g} mm/s" if measured else "NOT MEASURED")
    mon = bool(cfg.get("monitor_only"))
    rows.append(Row("T3", OK if measured else (WARN if mon else FAIL), "Travel time photo-eye -> reject cylinder",
                    f"{travel:.2f} s from {src}" + ("" if measured else
                    ". With the San_test ladder Y0 fires the moment M1 is written, so this number IS the reject timing."), "SAVED"))
    speed = float(cfg.get("conveyor_mm_s") or 0) or (float(cfg.get("inspection_to_reject_mm") or 0) / travel if measured else 0)

    # ---- how sure is the travel time? (the pusher only forgives reject_tolerance_s)
    tc = settings.get("travel_calibration") or {}
    sc = settings.get("speed_calibration") or {}
    if tc.get("runs", 0) >= 3:
        spread, basis = float(tc["spread_pct"]), f"{tc['runs']} stopwatch runs (travel_calibration)"
    elif sc.get("spread_pct") is not None:
        spread, basis = float(sc["spread_pct"]), "saved speed calibration"
    else:
        spread, basis = float(cfg.get("speed_tolerance_pct", 10.0)), "default tolerance - NO calibration saved"
    unc = travel * spread / 100.0 if measured else float("inf")
    tol = float(cfg["reject_tolerance_s"])
    good = measured and unc <= tol
    rows.append(Row("T4", OK if good else (WARN if measured else INFO), "Travel time repeatability vs late limit",
                    (f"+-{spread:g}% of {travel:.2f} s = +-{unc:.2f} s (about +-{unc * speed:.0f} mm) from {basis}; the cylinder "
                     f"forgives {tol:g} s (+-{tol * speed:.0f} mm, bottle {BOTTLE_MM:g} mm)"
                     + ("" if good else ". Too loose: run `--travel 8` (stopwatch) and fix the belt speed / sensor until the spread is smaller."))
                    if measured else "travel not measured", "SAVED" if "NO calibration" not in basis else "ESTIMATE"))

    # ---- one bottle's timeline
    delay = float(cfg.get("capture_delay_s") or 0)
    window = float(cfg["inspect_window_s"])
    frames = int(cfg["inspect_frames"])
    lead = MC.reject_lead_s(cfg)
    trig_gap = 0.05 + SERIAL_TXN_S                      # poll period + one transaction: how late Python can see M2
    win_open, win_close = delay, delay + window
    ready = win_close + MC.AI_ALLOWANCE_S
    send_at = travel - lead
    y0_on = send_at + WRITE_PATH_S + (0.0 if san else lead)
    y0_off = y0_on + MC.reject_pulse_s(cfg)
    timeline = [
        (0.0, "X0 photo-eye sees the bottle"),
        (SCAN_S, "PLC sets M2 (trigger)"),
        (SCAN_S + trig_gap, "Python has seen M2 (frame window may open; +- this is trigger_unc)"),
        (win_open, "frame window opens (capture_delay_s)"),
        (win_close, f"frame window closes ({frames} frames over {window:g} s)"),
        (ready, "verdict ready (window + AI_ALLOWANCE_S 0.4 s)"),
        (send_at, "REJECT: M1 written (PASS: M0 is written as soon as the verdict is ready)"),
        (y0_on, "Y0 ON (san_test: at once; M1 write path ~0.15 s)" if san else "Y0 ON (after the ladder's T0)"),
        (y0_off, f"Y0 OFF ({MC.reject_pulse_s(cfg):g} s pulse)"),
    ]
    timeline.sort()
    fit = MC.fit_problem(cfg)
    slack = send_at - ready
    margin = float(cfg.get("timing_margin_s", TR.TRACKING_DEFAULTS["timing_margin_s"]))
    st = OK if (measured and not fit and slack >= margin) else (FAIL if fit else WARN)
    rows.append(Row("T5", st if measured else INFO, "Verdict ready before the REJECT must go out",
                    (f"verdict ready at {ready:.2f} s, REJECT written at {send_at:.2f} s: slack {slack:.2f} s "
                     f"(needs >= {margin:g} s)") if measured else "travel unknown", "ESTIMATE"))
    rows.append(Row("T6", OK if (not measured or WRITE_PATH_S <= tol) else WARN, "M1 write path vs late limit",
                    f"Y0 comes ON about {WRITE_PATH_S:.2f} s after the scheduled time (precondition read + write at 9600 baud); "
                    f"late limit {tol:g} s. WATCH mode measures the real value.", "ESTIMATE"))
    if san and measured:
        mm = MC.reject_pulse_s(cfg) * speed
        rows.append(Row("T7", INFO, "Pusher stroke window",
                        f"Y0 is ON {MC.reject_pulse_s(cfg):g} s = the bottle moves {mm:.0f} mm during the pulse; centre the pulse on the "
                        f"bottle by trimming reject_travel_s (+/-) until the bottle is pushed square.", "ESTIMATE"))

    # ---- frames and AI
    spacing_s = window / max(1, frames - 1) if frames > 1 else window
    rows.append(Row("T8", OK if spacing_s >= AI_FRAME_S else WARN, "Frame spacing vs AI time per frame",
                    f"{frames} frames over {window:g} s = one every {spacing_s:.2f} s; the models need ~{AI_FRAME_S:g} s per frame"
                    + ("" if spacing_s >= AI_FRAME_S else ": scoring falls behind and the verdict is late"), "ESTIMATE"))

    # ---- bottle spacing (one-bottle handshake: M2 stays ON until the answer is written)
    if measured and speed:
        need_pass = ready
        need_rej = send_at
        sp = spacing_mm or float(settings.get("bottle_spacing_mm") or 0)
        detail = (f"M2 stays ON until the answer is written, and a bottle at X0 meanwhile gets NO trigger (NOT INSPECTED): after a "
                  f"PASS bottle the next must be >= {need_pass:.1f} s ({need_pass * speed:.0f} mm) behind, after a REJECT bottle "
                  f">= {need_rej:.1f} s ({need_rej * speed:.0f} mm). Max rate about {60 / max(need_pass, 0.1):.0f} bottles/min "
                  f"if every bottle is good.")
        status = INFO
        if sp:
            ok = sp >= need_rej * speed
            status = OK if ok else (WARN if sp >= need_pass * speed else FAIL)
            detail += f" Your spacing {sp:g} mm = {sp / speed:.1f} s -> " + ("enough." if ok else
                      "enough after a PASS but NOT after a REJECT: the bottle behind a rejected one is not inspected."
                      if status == WARN else "too close even for PASS bottles.")
        rows.append(Row("T9", status, "Minimum bottle spacing (one-bottle handshake)", detail, "ESTIMATE"))

    # ---- PC heartbeat / fault (san_test)
    if san:
        hb_on = bool(settings.get("plc_heartbeat"))
        lt1_s = lt["T1"] or 0
        ok = hb_on and lt1_s >= 2 * HEARTBEAT_PERIOD_S + 0.5
        rows.append(Row("T10", OK if ok else FAIL, "PC heartbeat vs the ladder's heartbeat-lost timer",
                        f"plc_heartbeat {'ON' if hb_on else 'OFF'}, pulse every {HEARTBEAT_PERIOD_S:g} s, ladder T1 = {lt1_s:g} s "
                        f"(one missed pulse is tolerated). With the heartbeat OFF the ladder sets M3 {lt1_s:g} s after the line starts.",
                        "SAVED"))
        dev = str(settings.get("estop_device") or "").upper()
        halt = dev == "M3" and bool(settings.get("estop_active_high"))
        rows.append(Row("T11", OK if halt else WARN, "Software halts when the PLC raises M3 (system fault)",
                        f"estop_device = '{dev or '-'}', estop_active_high = {bool(settings.get('estop_active_high'))}. "
                        + ("" if halt else "Set estop_device M3 + estop_active_high true (--apply does it). The ladder itself does NOT "
                           "stop the belt on M3, so without this the line keeps running through a heartbeat / contactor fault."), "SAVED"))
    cyc = MC.reject_cycle_s(cfg)
    rows.append(Row("T12", INFO, "Stuck-reject timeout", f"REJECT_NOT_OBSERVED if M1 is still ON {cyc + STUCK_EXTRA_S:.1f} s after it was written "
                    f"(cycle {cyc:g} s + {STUCK_EXTRA_S:g} s)", "SAVED"))
    rows.append(Row("T13", INFO, "Trigger detection delay",
                    f"Python sees M2 up to ~{trig_gap * 1000:.0f} ms after the PLC set it (poll 50 ms + one transaction); it records the "
                    f"+- on every bottle (trigger_unc_ms).", "ESTIMATE"))
    rows.append(Row("T14", OK if mon else INFO, "Monitor-only",
                    "ON: every bottle is judged and recorded, REJECT is answered M0, Y0 is never fired. Switch it OFF only after T1-T11 are OK."
                    if mon else "OFF: a REJECT fires the cylinder", "SAVED"))
    other = [p for p in (MC.safety_problem(cfg), MC.timing_problem(cfg), TR.calibration_problem(cfg)) if p]
    rows.append(Row("T15", FAIL if other else OK, "Software start check (machine_cycle.line_problems)",
                    "; ".join(other) if other else "no problem found (camera-station layout is checked at Start)", "SAVED"))

    wanted = {"plc_ladder": kind, "plc_t0_s": lt["T0"], "plc_t1_s": lt["T1"]}
    if san:
        wanted.update(plc_heartbeat=True, estop_device="M3", estop_active_high=True)
    return rows, timeline, {k: v for k, v in wanted.items() if v is not None}


def render(rows, timeline, title="TIMING CHECK") -> str:
    out = [title, "=" * len(title), "", "ONE BOTTLE (seconds after the photo-eye X0)"]
    out += [f"  {t:7.2f} s   {what}" for t, what in timeline]
    out += ["", "RESULTS"]
    for r in rows:
        out.append(f"  {r.status:<5} {r.rid:<4}{r.title}  [{r.source}]\n              {r.detail}")
    c = {k: sum(1 for r in rows if r.status == k) for k in (OK, WARN, FAIL, INFO)}
    out += ["", f"  OK {c[OK]}   WARN {c[WARN]}   FAIL {c[FAIL]}   INFO {c[INFO]}",
            "  " + ("NOT READY: fix the FAIL lines before switching monitor-only off." if c[FAIL] else
                    "NO FAIL. Fix the WARN lines, then run --watch on the real machine to measure them." if c[WARN] else
                    "READY on paper. Confirm on the machine with --watch.")]
    return "\n".join(out)


# ============================================================================================== APPLY
def app_running() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cmd = p.info.get("cmdline") or []
            if len(cmd) >= 2 and "python" in Path(cmd[0]).name.lower() and Path(cmd[-1]).name == "gui.py":
                return True
    except Exception:                                                  # noqa: BLE001 - psutil is optional
        pass
    return False


def apply_settings(wanted: dict, path: Path, force: bool = False) -> str:
    if app_running() and not force:
        return "NOT written: NexaInspect (gui.py) is running and would overwrite settings.json. Close it, or use --force."
    cur = json.loads(path.read_text())
    if "plc_ladder" not in cur:
        return ("NOT written: settings.json has no plc_ladder, so the plan above assumed 'final_year'. Set plc_ladder "
                "(\"san_test\" or \"final_year\") to the ladder that is in the PLC first, then run --apply again.")
    changes = {k: (cur.get(k), v) for k, v in wanted.items() if cur.get(k) != v}
    if not changes:
        return "settings.json already matches the ladder: nothing to change."
    backup = path.with_name(f"{path.name}.bak_timing_{time.strftime('%Y%m%d-%H%M%S')}")      # never overwrite an older backup
    shutil.copy2(path, backup)
    cur.update(wanted)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    tmp.replace(path)
    return "settings.json updated (backup settings.json.bak_timing):\n" + "\n".join(
        f"    {k}: {a!r} -> {b!r}" for k, (a, b) in changes.items())


# ============================================================================================== PLC (read-only)
def _service(settings: dict):
    from plc import PLCClient, PLCService, SerialTransport
    svc = PLCService(PLCClient(SerialTransport(str(settings.get("plc_com") or ""), int(settings.get("plc_baud", 9600)),
                                               str(settings.get("plc_format", "7E1"))), station=int(settings.get("plc_station", 1)),
                               timeout=1.0), poll_s=0.05, status_period_s=1.0)
    svc.start()
    svc.connect()
    return svc


def watch(settings: dict, seconds: float) -> str:
    """READ ONLY: sample X0 M2 M0 M1 Y0 Y1 M3 and print every bottle's measured timeline."""
    devs = ["X0", "M2", "M0", "M1", "Y0", "Y1", "M3"]
    svc = _service(settings)
    ev, last, t_start, n = [], {}, time.perf_counter(), 0
    try:
        print(f"watching {seconds:g} s - run bottles / a few rejects now (nothing is written) ...")
        while time.perf_counter() - t_start < seconds:
            t_req = time.perf_counter()
            s = svc.sample(bits=devs)
            t = (t_req + time.perf_counter()) / 2 - t_start
            n += 1
            for d in devs:
                v = int(bool(s.get(d)))
                if last.get(d) is not None and v != last[d]:
                    ev.append((t, d, v))
                last[d] = v
    finally:
        svc.stop()
    res = (time.perf_counter() - t_start) / max(1, n)
    out = [f"WATCH {seconds:g} s: {n} samples, resolution {res * 1000:.0f} ms per sample (every time below is +- that)"]
    xr = [t for t, d, v in ev if d == "X0" and v == 1]
    if not xr:
        out.append("no bottle passed the photo-eye (X0 never rose).")
    for i, x in enumerate(xr):
        nxt = xr[i + 1] if i + 1 < len(xr) else 1e9
        rel = [(t - x, d, v) for t, d, v in ev if x - res <= t < nxt]
        out.append(f"bottle {i + 1} at {x:7.2f} s" + (f"   (next bottle {nxt - x:.1f} s later)" if nxt < 1e8 else ""))
        out += [f"    +{t:6.2f} s  {d:<3}{'ON ' if v else 'off'}" for t, d, v in rel]
        y_on = [t for t, d, v in rel if d == "Y0" and v == 1]
        y_off = [t for t, d, v in rel if d == "Y0" and v == 0]
        m1 = [t for t, d, v in rel if d == "M1" and v == 1]
        if y_on and y_off:
            out.append(f"    -> Y0 pulse {y_off[0] - y_on[0]:.2f} s (ladder T0 {settings.get('plc_t0_s')} s)"
                       + (f"; Y0 came ON {y_on[0] - m1[0]:.2f} s after M1" if m1 else ""))
    if any(d == "M3" and v == 1 for _, d, v in ev):
        out.append("!! M3 (system fault) was raised during the watch: heartbeat lost / contactor feedback missing.")
    return "\n".join(out)


def travel(settings: dict, runs: int) -> tuple:
    """Stopwatch: X0 rising edge (read from the PLC) -> the operator presses Enter when the bottle is at the cylinder."""
    svc = _service(settings)
    times: list = []
    try:
        print(f"{runs} runs: put a bottle on the belt; press ENTER at the moment it is in front of the reject cylinder.")
        for i in range(runs):
            print(f"  run {i + 1}/{runs}: waiting for the photo-eye ...", flush=True)
            prev = int(bool(svc.sample(bits=["X0"])["X0"]))
            while True:
                t_req = time.perf_counter()
                cur = int(bool(svc.sample(bits=["X0"])["X0"]))
                if cur and not prev:
                    t_x0 = (t_req + time.perf_counter()) / 2
                    break
                prev = cur
            input("      bottle sensed - press ENTER when it is at the cylinder ")
            times.append(time.perf_counter() - t_x0)
            print(f"      {times[-1]:.2f} s")
    finally:
        svc.stop()
    return summarise_travel(times)


def summarise_travel(times: list) -> tuple:
    mean = statistics.mean(times)
    spread = (max(times) - min(times)) / 2 / mean * 100 if len(times) > 1 else 0.0
    rec = {"runs": len(times), "times_s": [round(t, 3) for t in times], "mean_s": round(mean, 3),
           "spread_pct": round(spread, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "method": "stopwatch: X0 edge read from the PLC, operator Enter at the cylinder"}
    return (f"mean travel {mean:.2f} s, spread +-{spread:.1f}% over {len(times)} runs "
            f"(operator reaction time ~0.2 s is inside this number)"), rec


# ============================================================================================== main
def selftest():
    s = json.loads((ROOT / "settings.json").read_text()) if (ROOT / "settings.json").exists() else {}
    base = dict(s, plc_ladder="san_test", plc_t0_s=0.3, plc_t1_s=3.0, plc_heartbeat=True, estop_device="M3",
                estop_active_high=True, inspection_to_reject_mm=1000.0, conveyor_mm_s=100.0, reject_travel_s=0.0,
                inspect_window_s=3.0, inspect_frames=10, capture_delay_s=0.0, monitor_only=False, speed_calibration=None,
                travel_calibration={"runs": 8, "spread_pct": 1.0})
    st = lambda d, **k: {r.rid: r.status for r in analyse(dict(d, **k))[0]}          # noqa: E731
    r = st(base)
    assert r["T1"] == r["T2"] == r["T3"] == r["T4"] == r["T5"] == r["T8"] == r["T10"] == r["T11"] == OK, r
    assert st(base, plc_ladder="final_year")["T1"] in (WARN, OK)                   # depends on which project was saved last
    assert st(base, plc_ladder="final_year")["T2"] == FAIL                         # settings say T0 0.3, that ladder has K50
    assert st(base, plc_t0_s=0.75)["T2"] == FAIL                                   # ladder has K3
    assert st(base, plc_heartbeat=False)["T10"] == FAIL
    assert st(base, estop_device="")["T11"] == WARN
    assert st(base, inspect_window_s=9.8)["T5"] == FAIL                            # verdict later than the REJECT must go out
    assert st(base, travel_calibration={"runs": 8, "spread_pct": 10.0})["T4"] == WARN
    r0 = st(base, conveyor_mm_s=0.0, inspection_to_reject_mm=0.0)
    assert r0["T3"] == FAIL and st(base, conveyor_mm_s=0.0, inspection_to_reject_mm=0.0, monitor_only=True)["T3"] == WARN
    rows, tl, wanted = analyse(base, spacing_mm=300.0)
    assert {x.rid: x.status for x in rows}["T9"] == FAIL and wanted["estop_device"] == "M3" and wanted["plc_t0_s"] == 0.3
    assert "ONE BOTTLE" in render(rows, tl) and tl == sorted(tl)
    msg, rec = summarise_travel([9.4, 9.6, 9.5])
    assert rec["runs"] == 3 and abs(rec["mean_s"] - 9.5) < 1e-6 and 1.0 < rec["spread_pct"] < 1.1, rec
    tmp = Path(tempfile.mkdtemp(prefix="tc_")) / "settings.json"
    tmp.write_text(json.dumps({"plc_t0_s": 1.0, "keep": 1}))
    assert "no plc_ladder" in apply_settings(wanted, tmp, force=True)                # never guess which ladder is in the PLC
    tmp.write_text(json.dumps({"plc_ladder": "san_test", "plc_t0_s": 1.0, "keep": 1}))
    assert "updated" in apply_settings(wanted, tmp, force=True) and json.loads(tmp.read_text())["keep"] == 1
    assert "nothing to change" in apply_settings(wanted, tmp, force=True)
    print("ok  timing_check: plan arithmetic (ladder vs settings, fit, spacing, heartbeat, M3 halt), stopwatch summary, "
          "apply with backup; no hardware touched")


def main(argv: list):
    if "--selftest" in argv:
        return selftest()
    sp = ROOT / "settings.json"
    settings = json.loads(sp.read_text())
    opt = lambda f, cast, default=None: cast(argv[argv.index(f) + 1]) if f in argv else default      # noqa: E731
    lines = []
    if "--watch" in argv:
        lines.append(watch(settings, opt("--watch", float, 60.0)))
    elif "--travel" in argv:
        msg, rec = travel(settings, opt("--travel", int, 8))
        lines.append(msg)
        if "--apply" in argv:
            lines.append(apply_settings({"reject_travel_s": rec["mean_s"], "travel_calibration": rec}, sp, "--force" in argv))
    else:
        rows, tl, wanted = analyse(settings, opt("--spacing", float, 0.0))
        lines.append(render(rows, tl))
        if "--apply" in argv:
            lines.append("\nAPPLY\n" + apply_settings(wanted, sp, "--force" in argv))
        any_fail = any(r.status == FAIL for r in rows)
    text = "\n".join(lines)
    print(text)
    out = ROOT / "reports" / "timing"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"timing_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    f.write_text(text, encoding="utf-8")
    print(f"\nreport saved: {f}")
    return 1 if "--watch" not in argv and "--travel" not in argv and any_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
