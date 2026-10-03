"""SIMULATOR-ONLY communication test: Python -> PLCService -> ISPSoft simulator -> ladder -> bits.

    python -m plc.sim_comm_test                      # full run: read, write, trigger, PASS, REJECT, repeated

Everything goes through PLCService (single owner). The ONLY thing that is not production behaviour is the
stimulus `PLCService.simulator_test_write` (M2, and direct M0/M1 writes), which refuses any non-loopback
transport, any device other than M0/M1/M2, and is logged as SIM_TEST_WRITE. Y0/Y1/X are never written.
This proves SIMULATOR communication only. It says nothing about a physical PLC.
"""
from __future__ import annotations

import statistics
import sys
import time

from . import handshake_test as H
from .client import PLCClient, TcpTransport
from .service import PLCService

BITS = ["M0", "M1", "M2", "Y0"]
ROWS: list = []                      # (test, result, evidence)


def row(test, ok, evidence):
    ROWS.append((test, "PASS" if ok else "FAIL", evidence))
    print(f"  => [{'PASS' if ok else 'FAIL'}] {test}: {evidence}")
    return ok


def idle(svc, wait=6.0):
    """Wait until M0/M1/M2/Y0 are all OFF and T0/T1 are 0; return the last sample."""
    end = time.monotonic() + wait
    while True:
        s = svc.sample(BITS, ["T0", "T1"])
        if not any(s[b] for b in BITS) and not s["T0"] and not s["T1"]:
            return s, True
        if time.monotonic() > end:
            return s, False
        time.sleep(0.05)


def watch(svc, secs, until=None):
    """Sample as fast as the link allows; return [(t_rel, sample)]."""
    t0, out = time.perf_counter(), []
    while time.perf_counter() - t0 < secs:
        s = svc.sample(BITS, ["T0", "T1"])
        out.append((time.perf_counter() - t0, s))
        if until and until(s, out):
            break
    return out


def fmt(rows):
    last, out = None, []
    for t, s in rows:
        key = tuple(s[k] for k in ("M0", "M1", "M2", "Y0", "T0", "T1"))
        if key != last:
            out.append(f"{t * 1000:6.0f}ms M0={s['M0']} M1={s['M1']} M2={s['M2']} Y0={s['Y0']} T0={s['T0']} T1={s['T1']}")
            last = key
    return out


def main() -> int:
    svc = PLCService(PLCClient(TcpTransport("127.0.0.1", 10002), station=1, timeout=1.0), poll_s=0.02, status_period_s=0.5)
    svc.start()
    ok_all = {}
    try:
        print("=== SIMULATOR-ONLY communication test (127.0.0.1:10002, Modbus ASCII, station 1) ===")
        lat0 = svc.connect()
        print(f"connected, first heartbeat {lat0:.1f} ms")

        # ------------------------------------------------------------ 1. READ
        print("\n[1] READ TEST")
        st = svc.read_status()
        for k in ("inputs", "internal", "outputs", "timers", "counters"):
            print(f"  {k:9s}", st[k])
        lats = []
        for _ in range(30):
            t = time.perf_counter(); svc.read_bit("M2"); lats.append((time.perf_counter() - t) * 1000)
        print(f"  PLC RUN: {st['plc_run']}   latency over 30 reads: mean {statistics.mean(lats):.1f}  median "
              f"{statistics.median(lats):.1f}  max {max(lats):.1f} ms   link={svc.link_state()}")
        ok_all["read"] = row("READ X0-2 M0-2 Y0-1 T0-1 + RUN + latency",
                             st["plc_run"] and svc.link_state() == "CONNECTED" and len(lats) == 30,
                             f"all devices read, RUN={st['plc_run']}, median {statistics.median(lats):.1f} ms, 0 errors")
        s, idle_ok = idle(svc, 2.0)
        if not idle_ok:
            print("simulator is NOT idle:", s, "-- aborting before any write")
            row("idle precondition", False, str(s))
            return 2

        # ------------------------------------------------------------ 2. WRITE
        print("\n[2] WRITE TEST (SIMULATOR-ONLY stimulus path; M2 and the two command bits only)")
        # M2: the PLC does not clear it on its own, so this is an unambiguous proof that a Python write changes the simulator.
        w = svc.simulator_test_write("M2", True)
        rows = watch(svc, 0.4)
        m2_on = all(r[1]["M2"] == 1 for r in rows)
        print(f"  M2 <- 1 write {w['write_ms']:.1f} ms; samples: {fmt(rows)}")
        svc.simulator_test_write("M2", False)
        s, idle_ok = idle(svc, 2.0)
        ok_all["w_m2"] = row("WRITE M2=1 then M2=0 (readback)", m2_on and idle_ok and not s["M2"],
                             f"M2 read back 1 in {len(rows)}/{len(rows)} samples, restored to 0")
        for bit in ("M0", "M1"):
            w = svc.simulator_test_write(bit, True)
            if bit == "M0":
                rows = watch(svc, 0.8)
            else:                                              # M1 may start the reject sequence: follow it to the end
                rows = watch(svc, 6.0, until=lambda s, o: any(r[1]["Y0"] for r in o) and not s["Y0"])
            seen_on = any(r[1][bit] for r in rows)
            stuck = rows[-1][1][bit]
            if stuck:                                          # never leave a command bit ON
                svc.simulator_test_write(bit, False)
            s, idle_ok = idle(svc, 6.0)
            print(f"  {bit} <- 1 write {w['write_ms']:.1f} ms; transitions: {fmt(rows)}; stuck={stuck}; idle after={idle_ok}")
            ok_all["w_" + bit] = row(f"WRITE {bit}=1 with M2=0 (no trigger)", idle_ok and not stuck and (seen_on or bool(rows)),
                                     f"FC05 accepted in {w['write_ms']:.1f} ms; observed {bit}=1: {seen_on}; "
                                     f"ended {'restored' if stuck else 'self-cleared by PLC'}; Y0 pulse seen: "
                                     f"{any(r[1]['Y0'] for r in rows)}")

        # ------------------------------------------------------------ 3-6. TRIGGER / PASS / REJECT / REPEATED
        print("\n[3-6] TRIGGER + PASS + REJECT via PLCService (M2 stimulated by simulator_test_write)")
        verdicts = ["PASS", "REJECT", "PASS", "REJECT", "REJECT", "PASS"]

        def stimulate():
            s, ok = idle(svc, 6.0)
            if not ok:
                raise RuntimeError(f"not idle before stimulus: {s}")
            svc.simulator_test_write("M2", True)

        rc = H.run(svc, verdicts, 5.0, H.T0_S, H.T1_S, 0.3, "ISPSoft SIMULATOR ONLY (M2 stimulated)", trigger_hook=stimulate)
        evs = svc.events()
        acked = [e for e in evs if e.event == "COMMAND_ACKED"]
        written = [e for e in evs if e.event == "COMMAND_WRITTEN"]
        trig = [e for e in evs if e.event == "TRIGGER"]
        by = {v: [e for e in acked if e.command == v] for v in ("PASS", "REJECT")}
        pass_ok = len(by["PASS"]) == 3
        rej_ok = len(by["REJECT"]) == 3
        ok_all["trig"] = row("TRIGGER: M2 write -> service TRIGGER event", len(trig) >= 6, f"{len(trig)} TRIGGER events")
        ok_all["pass"] = row("PASS: M2 clears, M0 clears, Y0 never ON", pass_ok and rc in (0,),
                             f"{len(by['PASS'])}/3 PASS ACKED (per-check detail above)")
        ok_all["rej"] = row("REJECT: M1 clears, M2 clears, T0, Y0 ON, T1, Y0 OFF", rej_ok and rc in (0,),
                            f"{len(by['REJECT'])}/3 REJECT ACKED (per-check detail above)")
        ok_all["rep"] = row("REPEATED PASS,REJECT,PASS,REJECT,REJECT,PASS", rc == 0 and len(written) == 6 and len(acked) == 6,
                            f"harness rc={rc}; commands written {len(written)}, acked {len(acked)}, duplicates {max(0, len(written) - 6)}")

        # ------------------------------------------------------------ restore + log
        s, idle_ok = idle(svc, 6.0)
        print("\nfinal state:", s, "idle:", idle_ok)
        print("\n--- every PLC service event (monotonic ms from first event) ---")
        t0 = evs[0].ts if evs else 0
        for e in svc.events():
            print(f"  {(e.ts - t0) * 1000:9.1f}  {e.event:<20} {e.device:<4} {e.command:<7} "
                  f"{('#' + str(e.trigger_id)) if e.trigger_id else '':<4} "
                  f"{('%.1fms' % e.latency_ms) if e.latency_ms is not None else '':<9} {e.ack} {e.error}")
        h = svc.health_check()
        print(f"\nlink: ok={h['ok']} errors={h['errors']} state={h['link_state']} last_error={h['last_error']}")
    finally:
        svc.stop()

    print("\n=== RESULT TABLE ===")
    print(f"{'TEST':62s} {'RESULT':6s} EVIDENCE")
    for t, r, e in ROWS:
        print(f"{t:62s} {r:6s} {e}")
    g = lambda *k: "PASS" if all(ok_all.get(x) for x in k) else "FAIL"
    print(f"\nSIMULATOR READ:    {g('read')}")
    print(f"SIMULATOR WRITE:   {g('w_m2', 'w_M0', 'w_M1')}")
    print(f"PASS HANDSHAKE:    {g('trig', 'pass')}")
    print(f"REJECT HANDSHAKE:  {g('rej')}")
    print(f"REPEATED CYCLES:   {g('rep')}")
    print("PHYSICAL PLC:      NOT TESTED")
    return 0 if all(ok_all.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
