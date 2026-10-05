"""Run the user's REAL ladder (read from the ISPSoft .isp) in a scan-by-scan simulator and test it against
what the inspection software needs. No PLC, no ISPSoft, no hardware: virtual time, runs in well under a second.

    python -m plc.ladder_sim                         # plc file/final_year/final_year.isp + settings.json
    python -m plc.ladder_sim path\\to\\project.isp [--t0 0.75 --t1 0.25 --estop X3 --mm 600 --speed 90]
    python -m plc.ladder_sim --selftest

What it answers: "if this ladder were in the PLC, would the software's PASS / REJECT handshake, timing and
bottle accounting work?" Every scenario is PASS / LIMIT / WARN / FAIL:

    PASS   the software contract holds
    LIMIT  works, but a known ladder limit applies (the software reports it, e.g. NOT INSPECTED)
    WARN   works for the software but is unsafe or unprotected (no E-stop input, no timeout, no interlock)
    FAIL   the software would misbehave on this ladder (wrong timing, no acknowledge, stuck bit)
    SKIP   cannot be simulated reliably (see the note)

Model (DVP-SS2, INFERRED from Delta's scan cycle and matched against the ISPSoft simulator on 2026-10-03):
networks run top to bottom each scan; X/M bits are read as they are at that point; a rising-edge contact
compares with the previous scan; TMR counts 100 ms units while its rung is true and resets when it is false;
SET / RST / OUT as named. The decoder (plc.ladder_check) reads each network's contacts but NOT the
series/parallel wiring: several contacts in one network are simulated as AND (series). A ladder that uses
OR branches (e.g. "X1 OR M10") is therefore flagged, and the scenarios it affects are SKIPped, not guessed.

The ladder is never written. docs/hardware/PLC_LADDER_REQUIREMENTS.md explains the requirements behind each check.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from plc import ladder_check as LC

SCAN_S = 0.005                       # virtual scan time
WATCH = ("X0", "X1", "X2", "X3", "M0", "M1", "M2", "M10", "M11", "Y0", "Y1", "T0", "T1")
PASS, LIMIT, WARN, FAIL, SKIP = "PASS", "LIMIT", "WARN", "FAIL", "SKIP"


class Sim:
    """Scan-by-scan interpreter of the decoded networks (virtual time)."""

    def __init__(self, nets):
        self.nets = nets
        self.bits: dict = {}
        self.prev: dict = {}
        self.acc: dict = {}                      # timer -> accumulated seconds
        self.preset: dict = {}                   # timer -> preset seconds
        self.count: dict = {}
        self.t = 0.0
        self.events: list = []                   # (t, device, value) for WATCH devices
        self._last: dict = {}
        for n in nets:
            for o in n.outputs:
                if o[0] == "TMR" and len(o) > 2:
                    self.preset[o[1]] = int(o[2]) * LC.TIMER_BASE_S
        self._log()

    def get(self, d: str) -> int:
        return int(self.bits.get(d, 0))

    def set(self, d: str, v: int):
        self.bits[d] = int(v)

    def _cond(self, n, now_x) -> bool:
        for kind, d in n.contacts:
            v = self.get(d)
            if kind == "RISE ":
                ok = bool(v) and not self.prev.get(d, 0)
            elif kind == "FALL ":
                ok = (not v) and bool(self.prev.get(d, 0))
            elif kind == "NOT ":
                ok = not v
            else:
                ok = bool(v)
            if not ok:
                return False
        return True

    def scan(self):
        for n in self.nets:
            c = self._cond(n, None)
            for o in n.outputs:
                op, dev = o[0], (o[1] if len(o) > 1 else "")
                if op == "SET" and c:
                    self.bits[dev] = 1
                elif op == "RST" and c:
                    self.bits[dev] = 0
                elif op == "OUT":
                    self.bits[dev] = int(c)
                elif op == "TMR":
                    if c:
                        self.acc[dev] = self.acc.get(dev, 0.0) + SCAN_S
                    else:
                        self.acc[dev] = 0.0
                    self.bits[dev] = int(c and self.acc[dev] >= self.preset.get(dev, 0) - 1e-9)
                elif op == "CNT":
                    if c and not self.prev.get(("cnt", n.id, dev), 0):
                        self.count[dev] = self.count.get(dev, 0) + 1
                    self.prev[("cnt", n.id, dev)] = int(c)
        for d in list(self.bits):
            if isinstance(d, str):
                self.prev[d] = self.bits[d]
        for d in WATCH:
            self.prev.setdefault(d, 0)
        self.t += SCAN_S
        self._log()

    def _log(self):
        for d in WATCH:
            v = self.get(d)
            if self._last.get(d) != v:
                self._last[d] = v
                self.events.append((round(self.t, 3), d, v))

    def run(self, seconds: float):
        for _ in range(int(round(seconds / SCAN_S))):
            self.scan()

    def pulse(self, d: str, seconds: float = 0.2):
        self.set(d, 1)
        self.run(seconds)
        self.set(d, 0)
        self.run(SCAN_S * 2)

    def times(self, dev: str, val: int, after: float = 0.0) -> list:
        return [t for t, d, v in self.events if d == dev and v == val and t >= after]


@dataclass
class Result:
    id: str
    title: str
    status: str
    detail: str
    impact: str = ""
    trace: list = field(default_factory=list)


def _fresh(nets, conveyor=True) -> Sim:
    s = Sim(nets)
    s.set("X3", 1)                                    # E-stop status contact, NC: 1 = healthy
    if conveyor:
        s.pulse("X1", 0.1)
    return s


def _trace(s: Sim, since=0.0, limit=40) -> list:
    return [f"{t:8.3f}s  {d:<4}{'ON ' if v else 'off'}" for t, d, v in s.events if t >= since][:limit]


def run_scenarios(nets, cfg: dict) -> list:
    t0 = float(cfg.get("plc_t0_s") or 0)
    t1 = float(cfg.get("plc_t1_s") or 0)
    pre = Sim(nets).preset                           # what the LADDER really has (not what settings.json says)
    lt0, lt1 = pre.get("T0", t0), pre.get("T1", t1)
    cycle = lt0 + lt1                                # one full reject cycle in the PLC
    R: list = []
    multi = [n for n in nets if len(n.contacts) > 1]
    uses = lambda dev: any(any(d == dev for _, d in n.contacts) or any(o[1:2] == (dev,) for o in n.outputs) for n in nets)  # noqa: E731

    if multi:
        R.append(Result("S0", "Ladder structure", WARN,
                        "network(s) " + ", ".join(str(n.id) for n in multi) + " have several contacts; series/parallel "
                        "wiring is not read, so they are simulated as AND (series)",
                        "results for the devices those networks drive may be wrong; check them in ISPSoft"))
    # ---------------------------------------------------------------- trigger
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    rose = s.times("M2", 1)
    held = s.get("M2")
    s.run(5.0)
    R.append(Result("S1", "Photo-eye triggers the inspection (X0 -> M2)",
                    PASS if rose and held and s.get("M2") else FAIL,
                    (f"M2 ON {rose[0] * 1000:.0f} ms after X0, stays ON until answered" if rose and s.get("M2")
                     else "M2 did not rise on an X0 pulse" if not rose else "M2 dropped without an answer"),
                    "every inspection starts from M2", _trace(s)))
    # ---------------------------------------------------------------- PASS
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    tp = s.t
    s.set("M0", 1)
    s.run(0.2)
    ok = not s.get("M2") and not s.get("M0")
    y0 = s.times("Y0", 1, tp)
    R.append(Result("S2", "PASS answer (M0) is acknowledged",
                    PASS if ok and not y0 else FAIL,
                    ("M0 and M2 cleared within 0.2 s, Y0 never ON, PASS counter C0 = "
                     f"{s.count.get('C0', '?')}") if ok and not y0 else
                    f"after M0: M2={s.get('M2')} M0={s.get('M0')} Y0 events={len(y0)}",
                    "the software waits for M0 and M2 to clear; otherwise PASS is reported NOT_ACKED", _trace(s, tp)))
    # ---------------------------------------------------------------- REJECT ack + cycle
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    tr = s.t
    s.set("M1", 1)
    s.run(0.1)
    ack = not s.get("M2")
    s.run(cycle * 1.5 + 3.0)
    R.append(Result("S3", "REJECT answer (M1) clears M2", PASS if ack else FAIL,
                    "M2 cleared within 0.1 s of M1" if ack else "M2 still ON after M1",
                    "the software takes 'M2 cleared' as the REJECT acknowledge (M1 may stay held)"))
    on, off = s.times("Y0", 1, tr), s.times("Y0", 0, tr)
    m1off = s.times("M1", 0, tr)
    t_on = (on[0] - tr) if on else None
    width = (off[0] - on[0]) if on and off else None
    stuck = [d for d in ("M1", "M2", "Y0", "T0", "T1") if s.get(d)]
    if not on:
        R.append(Result("S4", "REJECT cycle: Y0 pulse", FAIL, "Y0 never turned ON after M1",
                        "no physical reject", _trace(s, tr)))
    else:
        bad = []
        if t0 and abs(t_on - t0) > 0.2:
            bad.append(f"Y0 starts {t_on:.2f} s after M1 but settings plc_t0_s = {t0:g} s")
        if t1 and width is not None and abs(width - t1) > 0.2:
            bad.append(f"Y0 stays ON {width:.2f} s but settings plc_t1_s = {t1:g} s")
        if width is not None and width < 0.05:
            bad.append(f"Y0 is ON only {width * 1000:.0f} ms (one-scan flash: the cylinder will not stroke)")
        R.append(Result("S4", "REJECT cycle timing matches settings.json",
                        FAIL if bad else PASS,
                        "; ".join(bad) if bad else f"Y0 ON {t_on:.2f} s after M1 for {width:.2f} s (T0 {t0:g} s, T1 {t1:g} s)",
                        "REJECT is sent at trigger + travel - T0; a wrong T0 ejects the wrong bottle (or nothing)",
                        _trace(s, tr)))
        R.append(Result("S5", "Exactly one Y0 pulse per REJECT, and the cycle ends clean",
                        PASS if len(on) == 1 and not stuck else FAIL,
                        f"{len(on)} Y0 pulse(s); M1 released {(m1off[0] - tr):.2f} s after M1" if len(on) == 1 and not stuck
                        else f"{len(on)} pulse(s); still ON after the cycle: {', '.join(stuck) or '-'}",
                        "a stuck M1 or a repeated pulse would block or repeat rejects"))
    # ---------------------------------------------------------------- masking / one-bottle handshake
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    s.set("M1", 1)
    s.run(max(lt0, 0.5) * 0.5)
    n_before = len(s.times("M2", 1))
    s.pulse("X0", 0.2)
    masked = len(s.times("M2", 1)) == n_before
    R.append(Result("S6", "A bottle arriving during a reject cycle still gets a trigger",
                    LIMIT if masked else PASS,
                    (f"no trigger: M2 is held reset for the whole reject cycle (T0 + T1 = {cycle:g} s in the ladder). The software "
                     "records this bottle as NOT INSPECTED") if masked else "second trigger raised",
                    "continuous production: bottles closer than T0 + T1 after a reject are not inspected"))
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    n_before = len(s.times("M2", 1))
    s.run(1.0)
    s.pulse("X0", 0.2)
    R.append(Result("S7", "A second bottle before the first is answered raises a second trigger",
                    LIMIT if len(s.times("M2", 1)) == n_before else PASS,
                    "no: M2 is still ON for the first bottle (one-bottle handshake); the software reports NOT INSPECTED"
                    if len(s.times("M2", 1)) == n_before else "yes",
                    "the answer to a REJECT is held until its dispatch time, so spacing must exceed the travel time"))
    # ---------------------------------------------------------------- conveyor
    if any(n.id for n in nets if n.has_out("SET", "Y1") and len(n.contacts) > 1):
        R.append(Result("S8", "Conveyor start / stop", SKIP, "Y1 is driven by a network with several contacts (OR branches "
                        "not simulated)", "check start/stop on the real PLC"))
    else:
        s = Sim(nets)
        s.set("X3", 1)
        s.pulse("X1", 0.1)
        started = s.get("Y1")
        s.pulse("X2", 0.1)
        R.append(Result("S8", "Conveyor: X1 start latches Y1, X2 stop releases it",
                        PASS if started and not s.get("Y1") else FAIL,
                        "Y1 latched by X1 and released by X2" if started and not s.get("Y1")
                        else f"after X1 Y1={started}; after X2 Y1={s.get('Y1')}", "the PLC owns the conveyor"))
        s = Sim(nets)
        s.set("X3", 1)
        s.pulse("M10", 0.3)
        if s.get("Y1"):
            s.pulse("M11", 0.3)
        R.append(Result("S9", "Machine-page START / STOP test buttons (M10 / M11) move the conveyor",
                        PASS if s.times("Y1", 1) and not s.get("Y1") else WARN,
                        "M10 starts and M11 stops the conveyor" if s.times("Y1", 1) and not s.get("Y1")
                        else "M10 / M11 are written by the software but this ladder never reads them: the buttons do nothing",
                        "only matters with plc_operator_controls (commissioning aid)"))
    # ---------------------------------------------------------------- safety
    s = _fresh(nets)
    s.set("X3", 0)                                    # E-stop pressed (NC contact opens)
    s.run(0.2)
    est = str(cfg.get("estop_device") or "").upper()
    R.append(Result("S10", "E-stop status input stops the conveyor",
                    PASS if not s.get("Y1") else WARN,
                    "Y1 drops when X3 opens" if not s.get("Y1") else
                    "the ladder ignores X3: the conveyor keeps running (the hardware E-stop must cut power by itself)"
                    + ("" if est else "; no estop_device is configured either"),
                    "estop_device halts the SOFTWARE only"))
    s = Sim(nets)
    s.set("X3", 1)                                    # conveyor stopped
    s.pulse("X0", 0.2)
    s.set("M1", 1)
    s.run(cycle + 2.0)
    R.append(Result("S11", "Reject cylinder is blocked while the conveyor is stopped",
                    PASS if not s.times("Y0", 1) else WARN,
                    "no Y0 with Y1 off" if not s.times("Y0", 1) else "Y0 strokes with the belt stopped (no Y1 interlock)",
                    "machine safety / wrong-bottle risk (recommended interlock)"))
    s = _fresh(nets)
    s.pulse("X0", 0.2)
    s.run(30.0)
    R.append(Result("S12", "If the PC never answers, the ladder fails safe",
                    PASS if not s.get("M2") or s.times("Y1", 0, 1.0) else WARN,
                    "M2 cleared / belt stopped" if not s.get("M2") or s.times("Y1", 0, 1.0) else
                    "M2 stays ON for 30 s and nothing happens: the bottle passes uninspected if the PC stops",
                    "PC crash / freeze"))
    # ---------------------------------------------------------------- timing against the line settings
    import machine_cycle as MC
    lc = MC.line_settings(dict(cfg, plc_t0_s=lt0 or 0.01, plc_t1_s=lt1 or 0.01))    # the ladder's T0, as the PLC runs it
    bad = MC.timing_problem(lc)
    travel, measured = MC.travel_time(lc)
    R.append(Result("S13", "The ladder's T0 fits the belt (software start check)",
                    FAIL if bad else PASS,
                    bad or (f"ladder T0 {lt0:g} s < travel {travel:.2f} s" if measured else
                            f"ladder T0 {lt0:g} s is plausible, but distance / speed are not measured, so the real "
                            f"travel time is unknown"),
                    "the line refuses to start otherwise (machine_cycle.timing_problem)"))
    gap = cycle
    mm = float(cfg.get("conveyor_mm_s") or 0)
    R.append(Result("S14", "Minimum bottle spacing after a REJECT", LIMIT,
                    f"{gap:g} s (T0 + T1 in the ladder)" + (f" = {gap * mm:.0f} mm at {mm:g} mm/s" if mm else ""),
                    "closer bottles are NOT INSPECTED with this ladder"))
    return R


def summary(results) -> tuple:
    c = {k: sum(1 for r in results if r.status == k) for k in (PASS, LIMIT, WARN, FAIL, SKIP)}
    verdict = ("LADDER DOES NOT MEET THE SOFTWARE CONTRACT" if c[FAIL] else
               "LADDER WORKS WITH THE SOFTWARE (with the limits / warnings below)" if c[LIMIT] or c[WARN] else
               "LADDER MEETS THE SOFTWARE CONTRACT")
    return verdict, c


def simulate(path, cfg: dict) -> tuple:
    """(verdict, counts, results, networks) for an .isp file."""
    nets, _ = LC.load(Path(path))
    res = run_scenarios(nets, cfg)
    v, c = summary(res)
    return v, c, res, nets


def report(path, cfg: dict, detail: bool = False) -> str:
    v, c, res, nets = simulate(path, cfg)
    out = [f"ladder: {path}", f"settings used: T0 {cfg.get('plc_t0_s')} s, T1 {cfg.get('plc_t1_s')} s, "
           f"estop {cfg.get('estop_device') or '-'}", "", f"{v}",
           "  " + "  ".join(f"{k} {n}" for k, n in c.items()), ""]
    for r in res:
        out.append(f"{r.status:<5} {r.id:<4}{r.title}\n           {r.detail}\n           impact: {r.impact}")
        if detail and r.trace:
            out += ["             " + t for t in r.trace[:14]]
    return "\n".join(out)


def _settings_cfg(args: list) -> dict:
    cfg: dict = {}
    try:
        import dataset as D
        cfg = dict(D.load_settings())
    except Exception:                                                  # noqa: BLE001 - settings are optional
        pass
    for flag, key, cast in (("--t0", "plc_t0_s", float), ("--t1", "plc_t1_s", float), ("--estop", "estop_device", str),
                            ("--mm", "inspection_to_reject_mm", float), ("--speed", "conveyor_mm_s", float)):
        if flag in args:
            cfg[key] = cast(args[args.index(flag) + 1])
    return cfg


def selftest():
    def N(i, contacts, outs):
        n = LC.Net(i)
        n.contacts, n.outputs = contacts, outs
        return n
    saved = [N(1, [("", "X1")], [("SET", "Y1")]), N(2, [("", "X2")], [("RST", "Y1")]),
             N(3, [("RISE ", "X0")], [("SET", "M2")]),
             N(4, [("", "M0")], [("RST", "M2"), ("RST", "M0"), ("CNT", "C0", "9999")]),
             N(5, [("", "M1")], [("TMR", "T0", "150"), ("RST", "M2"), ("CNT", "C1", "9999")]),
             N(6, [("", "T0")], [("TMR", "T1", "50"), ("OUT", "Y0")]),
             N(7, [("", "T1")], [("RST", "M1"), ("RST", "M2")])]
    st = lambda res: {r.id: r.status for r in res}                     # noqa: E731
    # the saved ladder with the settings the software had (0.75 / 0.25): timing MISMATCH, everything else as documented
    r = st(run_scenarios(saved, {"plc_t0_s": 0.75, "plc_t1_s": 0.25}))
    assert r["S1"] == r["S2"] == r["S3"] == r["S5"] == r["S8"] == PASS, r
    assert r["S4"] == FAIL and r["S13"] == FAIL and r["S6"] == r["S7"] == LIMIT, r      # K150 = 15 s is not a travel time
    assert r["S9"] == r["S10"] == r["S11"] == r["S12"] == WARN, r
    # the same ladder with settings equal to its presets (15 s / 5 s): timing matches, but the line refuses to start
    r = st(run_scenarios(saved, {"plc_t0_s": 15.0, "plc_t1_s": 5.0}))
    assert r["S4"] == PASS and r["S13"] == FAIL, r
    # the stated contract (K15 / K5) with matching settings: all timing checks pass; Y0 pulse is T1 long
    ctr = [n if n.id not in (5, 6) else N(n.id, n.contacts, [(o if o[0] != "TMR" else (o[0], o[1], "15" if o[1] == "T0" else "5"))
                                                             for o in n.outputs]) for n in saved]
    res = run_scenarios(ctr, {"plc_t0_s": 1.5, "plc_t1_s": 0.5})
    r = st(res)
    assert r["S4"] == PASS and r["S5"] == PASS and r["S13"] == PASS, r
    v, c = summary(res)
    assert "LIMITS" in v.upper() or "LIMIT" in v.upper(), v
    # the old one-scan Y0 flash (01:36 ladder: T0 -> OUT Y0 + RST M1 in one rung) is caught
    flash = [n if n.id != 6 else N(6, [("", "T0")], [("TMR", "T1", "50"), ("OUT", "Y0"), ("RST", "M1"), ("RST", "M2")])
             for n in saved[:6]]
    r = st(run_scenarios(flash, {"plc_t0_s": 15.0, "plc_t1_s": 5.0}))
    assert r["S4"] == FAIL, r
    # a ladder with no reject cycle at all
    r = st(run_scenarios(saved[:5], {"plc_t0_s": 1.5, "plc_t1_s": 0.5}))
    assert r["S4"] == FAIL, r
    # several contacts in one network are flagged, not guessed
    r = st(run_scenarios(saved + [N(8, [("", "X1"), ("", "M10")], [("SET", "Y1")])], {"plc_t0_s": 15.0, "plc_t1_s": 5.0}))
    assert r["S0"] == WARN and r["S8"] == SKIP, r
    print("ok  ladder_sim: scan simulator on the saved ladder (timing mismatch caught, masking + one-bottle limits "
          "reported), contract ladder passes, Y0 flash and missing reject cycle caught, OR networks flagged")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    else:
        path = Path(a[0]) if a and not a[0].startswith("--") else LC.DEFAULT_ISP
        print(report(path, _settings_cfg(a), detail="--trace" in a))
