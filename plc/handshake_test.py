"""Handshake verification of the current machine contract, through PLCService.

    python -m plc.handshake_test --real --cycles PASS,REJECT,PASS --wait 60   # ISPSoft SIMULATOR
    python -m plc.handshake_test --fake                                       # FAKE PLC (harness self-check)

--real  needs ISPSoft + COMMGR + the DVP-SS2 simulator running in RUN mode. For every cycle YOU raise
        X0 in the simulator (the photoelectric sensor); this script never forces X0 or M2. It waits
        for M2 to appear, sends the command ONLY via PLCService (M0 = PASS, M1 = REJECT), then
        samples M0 M1 M2 Y0 Y1 and T0 T1 until the machine is idle again and checks:
            PASS   : M0 cleared by the PLC, M2 cleared, Y0 never ON
            REJECT : M1 cleared, M2 cleared, T0 runs, Y0 ON after ~T0, T1 runs, Y0 OFF after ~T1
        Results are SIMULATOR ONLY: they say nothing about a physical PLC's timing.
--fake  runs the same harness against FakeLadder (scaled timers), to test the harness itself.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from . import address_map as AM
from .client import PLCClient, TcpTransport
from .service import ACKED, PLCService

T0_S, T1_S = 1.5, 0.5                      # intended values (K15, K5 at 100 ms) -- from the user
WATCH_BITS = ["M0", "M1", "M2", "Y0", "Y1"]
WATCH_WORDS = ["T0", "T1", "C0", "C1"]


def observe_cycle(svc: PLCService, verdict: str, t_cmd: float, t0_s: float, t1_s: float, tail_s: float):
    """Sample until Y0 has completed a pulse (REJECT) or `tail_s` has elapsed (PASS)."""
    rows, y0_on, y0_off = [], None, None
    t_end = time.perf_counter() + (t0_s + t1_s + 1.5 if verdict == "REJECT" else tail_s)
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        smp = svc.sample(WATCH_BITS, WATCH_WORDS)
        rows.append((t - t_cmd, smp))
        if smp["Y0"] and y0_on is None:
            y0_on = t - t_cmd
        if y0_on is not None and not smp["Y0"] and y0_off is None:
            y0_off = t - t_cmd
        if verdict == "REJECT" and y0_off is not None:
            break
    return rows, y0_on, y0_off


def check(verdict, res, rows, y0_on, y0_off, t0_s, t1_s, tol):
    """Return a list of (name, passed, detail)."""
    out = [("PLC acknowledged (command bit + M2 cleared)", res.status == ACKED, f"{res.status} {res.detail}")]
    last = rows[-1][1] if rows else {}
    out.append(("M0 and M1 both OFF afterwards", not last.get("M0") and not last.get("M1"),
                f"M0={last.get('M0')} M1={last.get('M1')}"))
    out.append(("M2 OFF afterwards", not last.get("M2"), f"M2={last.get('M2')}"))
    if verdict == "PASS":
        out.append(("no reject actuation (Y0 never ON)", y0_on is None, f"Y0 first ON at {y0_on}"))
        out.append(("T0 never ran", not any(r[1]["T0"] for r in rows), ""))
    else:
        out.append(("T0 ran", any(r[1]["T0"] for r in rows), f"max T0 = {max((r[1]['T0'] for r in rows), default=0)}"))
        ok_on = y0_on is not None
        out.append(("Y0 turned ON", ok_on, f"{y0_on if y0_on is None else round(y0_on, 3)} s after the command"))
        out.append(("Y0 ON after about T0 (intended %.1f s)" % t0_s,
                    ok_on and abs(y0_on - t0_s) <= tol, f"observed {y0_on if y0_on is None else round(y0_on, 3)} s, tolerance {tol} s"))
        ok_off = y0_on is not None and y0_off is not None
        out.append(("T1 ran", any(r[1]["T1"] for r in rows), f"max T1 = {max((r[1]['T1'] for r in rows), default=0)}"))
        out.append(("Y0 turned OFF", ok_off, ""))
        if ok_off:
            pulse = y0_off - y0_on
            out.append(("Y0 pulse about T1 (intended %.1f s)" % t1_s, abs(pulse - t1_s) <= tol,
                        f"observed {pulse:.3f} s, tolerance {tol} s"))
    return out


def _stats(name, vals, fails=0):
    if not vals:
        return f"  {name:46s} no data   failures {fails}"
    v = sorted(vals)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return (f"  {name:46s} N={len(v):3d} mean {statistics.mean(v):8.1f}  median {statistics.median(v):8.1f}  "
            f"min {v[0]:8.1f}  max {v[-1]:8.1f}  p95 {q(.95):8.1f}  p99 {q(.99):8.1f} ms   failures {fails}")


def run(svc: PLCService, verdicts, wait_s, t0_s, t1_s, tol, label, trigger_hook=None) -> int:
    print(f"\n=== {label} :: {len(verdicts)} cycle(s): {','.join(verdicts)} ===")
    snap = svc.read_status()
    idle = {k: snap[k] for k in ("inputs", "internal", "outputs", "timers", "counters")}
    print("idle state before the run:", idle)
    if any(snap["internal"][d] for d in ("M0", "M1", "M2")) or any(snap["outputs"].values()):
        print("NOT IDLE: M0/M1/M2/Y0/Y1 already ON. Not starting; put the machine in a known idle state first.")
        return 2
    results, all_ok = [], True
    for i, verdict in enumerate(verdicts, 1):
        if trigger_hook:
            trigger_hook()
        else:
            print(f"\n[cycle {i}/{len(verdicts)}] waiting up to {wait_s:g}s for M2 -> raise X0 in the simulator now ...", flush=True)
        trig = svc.wait_for_trigger(wait_s)
        if trig is None:
            print("  no trigger (M2 never became 1): cycle skipped")
            all_ok = False
            continue
        pre = svc.sample(WATCH_BITS, WATCH_WORDS)
        t_seen = time.perf_counter()
        t_cmd = time.perf_counter()
        res = svc.submit_result(trig.id, verdict)
        rows, y0_on, y0_off = observe_cycle(svc, verdict, t_cmd, t0_s, t1_s, tail_s=t0_s + t1_s + 0.5)
        checks = check(verdict, res, rows, y0_on, y0_off, t0_s, t1_s, tol)
        ok = all(c[1] for c in checks)
        all_ok &= ok
        print(f"  trigger id {trig.id}  M2 seen; before command: {pre}")
        print(f"  {verdict}: status {res.status}; write->response {res.write_ms and round(res.write_ms, 1)} ms; "
              f"write->PLC ack {res.ack_ms and round(res.ack_ms, 1)} ms; command bit ever seen ON: {res.cmd_seen_on}")
        for name, passed, detail in checks:
            print(f"    [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
        results.append(dict(verdict=verdict, res=res, y0_on=y0_on, y0_off=y0_off, ok=ok, n_samples=len(rows),
                            sample_ms=(rows[-1][0] - rows[0][0]) / max(1, len(rows) - 1) * 1000 if len(rows) > 1 else None))
        time.sleep(0.3)
        if trigger_hook is None and svc.current_trigger() is not None:
            print("  note: M2 already set again (X0 still high / a new bottle)")
    # ---- summary
    print(f"\n--- summary ({label}) ---")
    n = len(results)
    print(f"cycles completed {n}/{len(verdicts)};  cycles passing every check {sum(r['ok'] for r in results)}")
    dup = [e for e in svc.events() if e.event == "COMMAND_WRITTEN"]
    print(f"commands written: {len(dup)} for {n} answered triggers (duplicates: {max(0, len(dup) - n)})")
    print(f"refused: {sum(e.event == 'COMMAND_REFUSED' for e in svc.events())}   "
          f"faults: {sum(e.event == 'FAULT' for e in svc.events())}")
    for v in ("PASS", "REJECT"):
        rr = [r for r in results if r["verdict"] == v]
        if rr:
            print(_stats(f"{v}: write request -> response", [r["res"].write_ms for r in rr if r["res"].write_ms]))
            print(_stats(f"{v}: write -> PLC ack (bit+M2 cleared)", [r["res"].ack_ms for r in rr if r["res"].ack_ms]))
    rj = [r for r in results if r["verdict"] == "REJECT" and r["y0_on"] is not None]
    if rj:
        print(_stats("REJECT: command -> Y0 ON (sampled)", [r["y0_on"] * 1000 for r in rj]))
        print(_stats("REJECT: Y0 ON -> Y0 OFF (sampled)", [(r["y0_off"] - r["y0_on"]) * 1000 for r in rj if r["y0_off"]]))
    print("(sampling resolution is the sample period; these are communication/observation timings, not machine timing)")
    return 0 if all_ok and n == len(verdicts) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--real", action="store_true", help="ISPSoft simulator on host:port (YOU raise X0 each cycle)")
    g.add_argument("--fake", action="store_true", help="FAKE PLC + FakeLadder")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=10002)
    ap.add_argument("--station", type=int, default=1)
    ap.add_argument("--cycles", default="PASS,REJECT,PASS,REJECT,REJECT,PASS")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for each M2")
    ap.add_argument("--tol", type=float, default=0.25, help="tolerance (s) when comparing observed to intended T0/T1")
    a = ap.parse_args(argv)
    verdicts = [v.strip().upper() for v in a.cycles.split(",") if v.strip()]
    assert all(v in AM.COMMAND_BITS for v in verdicts), "cycles must be PASS or REJECT"
    if a.fake:
        from .test_simulation import FakeLadder, FakePLC, _client
        fake = FakePLC(); lad = FakeLadder(fake, scale=1.0)
        svc = PLCService(_client(fake, timeout=0.5), poll_s=0.01, status_period_s=0.2)
        svc.start(); svc.connect()
        try:
            return run(svc, verdicts, 5.0, T0_S, T1_S, a.tol, "FAKE PLC (harness self-check, not evidence)",
                       trigger_hook=lad.trigger)
        finally:
            svc.stop(); lad.stop(); fake.close()
    svc = PLCService(PLCClient(TcpTransport(a.host, a.port), station=a.station, timeout=1.0),
                     poll_s=0.02, status_period_s=0.5)
    svc.start()
    try:
        svc.connect()
        return run(svc, verdicts, a.wait, T0_S, T1_S, a.tol, "ISPSoft SIMULATOR ONLY")
    finally:
        svc.stop()


if __name__ == "__main__":
    sys.exit(main())
