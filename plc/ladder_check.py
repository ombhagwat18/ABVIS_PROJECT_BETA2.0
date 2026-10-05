"""Read an ISPSoft ladder project (.isp) and check it against what this software needs from the PLC.

    python -m plc.ladder_check                                   # plc file/final_year/final_year.isp
    python -m plc.ladder_check "path/to/project.isp" [--t0 0.75] [--t1 0.25]
    python -m plc.ladder_check --selftest

READ-ONLY. The ladder belongs to the user; nothing here writes a project or talks to a PLC.

File format (VERIFIED on final_year.isp, ISPSoft 3.24): a binary header, then a raw-deflate stream
(zlib wbits=-15) of ISPSoft's text project. The header length is not fixed across saves (0xAA in the
2026-10-03 01:36 file, 264 bytes in the 2026-10-04 one), so the stream start is searched for.
Node TYPE codes as read from these files: 1 = NO contact, 2 = NC contact, 3 = rising-edge contact,
4 = falling-edge contact, 13 = OUT, 15 = SET, 16 = RST, 11 + 9 = API instruction (operands, SYMB).
Branch/link nodes (20, 29, 7, 12) are skipped: series/parallel structure inside one network is NOT
reconstructed, so a network's condition is reported as the list of its contacts. The requirement checks
below only rely on which contacts and which outputs a network has.

Each requirement is PRESENT / ABSENT / MISMATCH / INFO, and says which software feature depends on it.
docs/hardware/PLC_LADDER_REQUIREMENTS.md explains every one of them and gives the recommended rungs.
"""
from __future__ import annotations

import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ISP = Path(__file__).resolve().parent.parent / "plc file" / "final_year" / "final_year.isp"
CONTACT = {1: "", 2: "NOT ", 3: "RISE ", 4: "FALL "}
COIL = {13: "OUT", 15: "SET", 16: "RST"}
TIMER_BASE_S = 0.1                      # DVP-SS2 T0..T63: 100 ms base (INFERRED from Delta docs; matches simulator)


@dataclass
class Net:
    id: int
    contacts: list = field(default_factory=list)       # [(kind, device)]  kind: "", "NOT ", "RISE ", "FALL "
    outputs: list = field(default_factory=list)        # [("SET", "M2")] / [("TMR", "T0", "150")] / [("CNT", "C0", "9999")]

    def text(self) -> str:
        cond = " , ".join(f"{k}{d}" for k, d in self.contacts) or "(always)"
        outs = "; ".join(" ".join(o) for o in self.outputs)
        return f"net {self.id}: {cond} -> {outs}"

    def has_contact(self, dev, kinds=None) -> bool:
        return any(d == dev and (kinds is None or k in kinds) for k, d in self.contacts)

    def has_out(self, op, dev) -> bool:
        return any(o[0] == op and o[1] == dev for o in self.outputs)


def decompress(raw: bytes) -> str:
    for off in range(0, min(len(raw), 2048)):
        try:
            d = zlib.decompressobj(-15)
            t = d.decompress(raw[off:])
        except zlib.error:
            continue
        if d.eof and b"<NETWORK_START>" in t:
            return t.decode("latin-1")
    raise ValueError("no ISPSoft text project found in this file (not an ISPSoft .isp?)")


def parse(text: str) -> tuple:
    """(networks, device comments, ladder text) from the decompressed project text."""
    comments = {}
    m = re.search(r"\[DEVICE_CMT_START\](.*?)\[DEVICE_CMT_END\]", text, re.S)
    if m:
        for line in m.group(1).strip().splitlines():
            if "::" in line:
                k, v = line.split("::", 1)
                comments[k.strip()] = v.strip()
    nets = []
    for block in re.findall(r"<NETWORK_START>(.*?)<NETWORK_END>", text, re.S):
        nid = int(re.search(r"NET_ID=(\d+)", block).group(1))
        if re.search(r"NET_ACTIVE=FALSE", block):
            continue                                          # a disabled network does nothing in the PLC
        n = Net(nid)
        root = re.search(r"<ROOTLINK_START>(.*?)<ROOTLINK_END>", block, re.S)
        out = re.search(r"<OUTLINK_START>(.*?)<OUTLINK_END>", block, re.S)
        for part, is_root in ((root, True), (out, False)):
            if not part:
                continue
            pending = None
            for node in re.findall(r"\[LD_NODE\](.*?)\[END_LD_NODE\]", part.group(1), re.S):
                ty = int(re.search(r"TYPE=(\d+)", node).group(1))
                dev = (re.search(r"DEV_NAME=(\S*)", node) or [None, ""])[1] if "DEV_NAME=" in node else ""
                if ty in CONTACT and dev:
                    n.contacts.append((CONTACT[ty], dev))
                elif ty in COIL and dev:
                    n.outputs.append((COIL[ty], dev))
                elif ty == 11:
                    pending = re.findall(r"VAR_NAME=(\S+)", node)
                elif ty == 9:
                    symb = re.search(r"SYMB=(\S+)", node).group(1)
                    n.outputs.append((symb, *(pending or [])))
                    pending = None
        nets.append(n)
    return nets, comments


def load(path: Path) -> tuple:
    return parse(decompress(Path(path).read_bytes()))


# --------------------------------------------------------------------------------- requirements
def _timer(nets, t):
    for n in nets:
        for o in n.outputs:
            if o[0] == "TMR" and len(o) > 2 and o[1] == t:
                return n, o[2]
    return None, None


def check(nets, t0_s: float | None = None, t1_s: float | None = None, estop: str = "") -> list:
    """[(id, status, requirement, finding, needed by)]"""
    R = []

    def add(rid, ok, req, found, need, status=None):
        R.append((rid, status or ("PRESENT" if ok else "ABSENT"), req, found, need))

    def find(pred):
        return [n for n in nets if pred(n)]

    f = find(lambda n: n.has_contact("X0", ("RISE ",)) and n.has_out("SET", "M2"))
    add("R1", bool(f), "Photo-eye X0 rising edge -> SET M2 (inspection trigger)",
        f[0].text() if f else "no X0-rising -> SET M2 network", "every inspection (PLCService trigger)")
    f = find(lambda n: n.has_contact("M0") and n.has_out("RST", "M2") and n.has_out("RST", "M0"))
    add("R2", bool(f), "M0 (PASS answer) -> RST M2 and RST M0 in the same scan",
        f[0].text() if f else "missing", "PASS acknowledge (M0 and M2 cleared)")
    f = find(lambda n: n.has_contact("M1") and n.has_out("RST", "M2"))
    add("R3", bool(f), "M1 (REJECT answer) -> RST M2",
        f[0].text() if f else "missing", "REJECT acknowledge (M2 cleared; M1 may be held, HELD_UNTIL_DONE)")
    t0n, t0k = _timer(nets, "T0")
    y0 = find(lambda n: n.has_contact("T0") and n.has_out("OUT", "Y0"))
    t1n, t1k = _timer(nets, "T1")
    m1rst = find(lambda n: n.has_out("RST", "M1"))
    ok = bool(t0n and t0n.has_contact("M1") and y0 and t1n and m1rst)
    add("R4", ok, "Reject cycle in the PLC: M1 -> T0 (travel / lead) -> Y0 for T1 -> RST M1",
        "; ".join(x.text() for x in ([t0n] if t0n else []) + y0 + m1rst) or "missing",
        "physical reject (Python never writes Y0)")
    for rid, tn, k, want, name, why in (
            ("R5", "T0", t0k, t0_s, "T0 travel/lead delay",
             "REJECT dispatch = trigger + travel - T0: a wrong T0 hits the wrong bottle"),
            ("R6", "T1", t1k, t1_s, "T1 reject pulse",
             "pulse length and the 'stuck reject' timeout (T0 + T1 + 3 s); while M1 is held, triggers are masked")):
        if k is None:
            add(rid, False, f"{name} preset", "timer not found", "reject timing")
            continue
        sec = int(k) * TIMER_BASE_S
        if want is None:
            R.append((rid, "INFO", f"{name} preset", f"K{k} = {sec:.1f} s", "settings plc_t0_s / plc_t1_s must equal it"))
        else:
            same = abs(sec - want) < 0.05
            R.append((rid, "PRESENT" if same else "MISMATCH", f"{name} preset == settings.json ({want:g} s)",
                      f"K{k} = {sec:.1f} s in the ladder vs {want:g} s in settings.json",
                      why))
    f1 = find(lambda n: n.has_contact("X1") and n.has_out("SET", "Y1"))
    f2 = find(lambda n: n.has_contact("X2") and n.has_out("RST", "Y1"))
    add("R7", bool(f1 and f2), "Conveyor: X1 start -> SET Y1, X2 stop -> RST Y1",
        "; ".join(x.text() for x in f1 + f2) or "missing", "conveyor (the PLC owns Y1)")
    s = find(lambda n: n.has_contact("M10") and n.has_out("SET", "Y1"))
    p = find(lambda n: n.has_contact("M11") and n.has_out("RST", "Y1"))
    add("R8", bool(s and p), "Operator test bits: (X1 OR M10) -> SET Y1, (X2 OR M11) -> RST Y1",
        "; ".join(x.text() for x in s + p) or "M10 / M11 not used by any network",
        "Machine page START/STOP test buttons (plc_operator_controls) - without it they write bits nothing reads")
    dev = (estop or "X3").upper()
    e = find(lambda n: n.has_contact(dev) and (n.has_out("RST", "Y1") or n.has_out("RST", "Y0")))
    add("R9", bool(e), f"E-stop status input ({dev}, NC) stops the conveyor and blocks Y0",
        "; ".join(x.text() for x in e) or f"{dev} not used by any network",
        "estop_device halt + start check (the hardware E-stop must ALSO cut power on its own)")
    il = find(lambda n: n.has_out("OUT", "Y0") and n.has_contact("Y1"))
    add("R10", bool(il), "Y0 interlocked with the conveyor (no reject stroke while Y1 is OFF)",
        "; ".join(x.text() for x in il) or "Y0 has no Y1 condition", "machine safety (recommended)")
    mask = find(lambda n: n.has_contact("M1") and n.has_out("RST", "M2")) + \
        find(lambda n: n.has_contact("T1") and n.has_out("RST", "M2"))
    held = bool(mask and t0n and t0n.has_contact("M1"))
    R.append(("R11", "MISMATCH" if held else "PRESENT",
              "Triggers NOT masked during a reject cycle (M1 must not hold M2 reset for T0 + T1)",
              ("M1 is held for T0 + T1 and its rung resets M2 every scan: a bottle at X0 in that window gets no "
               "trigger (BOTTLE_UNTRIGGERED)") if held else "no masking found",
              "continuous production (bottle spacing < T0 + T1)"))
    to = find(lambda n: n.has_contact("M2") and any(o[0] == "TMR" for o in n.outputs))
    add("R12", bool(to), "Answer timeout: M2 ON longer than N s without an answer -> fail-safe (reject / stop)",
        "; ".join(x.text() for x in to) or "no timer on M2: if the PC dies the bottle passes uninspected",
        "fail-safe when the PC / software stops answering")
    hb = find(lambda n: n.has_contact("M20") and any(o[0] == "TMR" for o in n.outputs))
    add("R13", bool(hb), "PC heartbeat watchdog (PC toggles a bit, PLC stops the line if it stops)",
        "; ".join(x.text() for x in hb) or "none", "detect a crashed PC (software side not built either)")
    return R


def report(path: Path, t0_s=None, t1_s=None, estop="") -> str:
    nets, comments = load(path)
    lines = [f"ladder: {path}", "", "NETWORKS (as decoded; branch structure not reconstructed)"]
    lines += ["  " + n.text() for n in nets]
    lines += ["", "DEVICE COMMENTS (from the project)"] + [f"  {k:<5} {v}" for k, v in comments.items()]
    lines += ["", "REQUIREMENTS"]
    for rid, st, req, found, need in check(nets, t0_s, t1_s, estop):
        lines.append(f"  {rid:<4}{st:<9}{req}\n           found:  {found}\n           needed: {need}")
    return "\n".join(lines)


def selftest():
    def net(i, root, outs):
        r = "".join(f"[LD_NODE]\nTYPE={t}\nDEV_NAME={d}\n[END_LD_NODE]\n" for t, d in root)
        o = ""
        for x in outs:
            if x[0] in ("TMR", "CNT"):
                o += f"[LD_NODE]\nTYPE=11\n<VAR_NODE_S>\nVAR_NAME={x[1]}\nVAR_NAME={x[2]}\n<VAR_NODE_E>\n[END_LD_NODE]\n" \
                     f"[LD_NODE]\nTYPE=9\nSYMB={x[0]}\nDEV_NAME=\n[END_LD_NODE]\n[LD_NODE]\nTYPE=20\nDEV_NAME=\n[END_LD_NODE]\n"
            else:
                o += f"[LD_NODE]\nTYPE={ {'OUT': 13, 'SET': 15, 'RST': 16}[x[0]] }\nDEV_NAME={x[1]}\n[END_LD_NODE]\n"
        return (f"<NETWORK_START>\n<PROPERTIES_START>\nNET_ID={i}\nNET_ACTIVE=TRUE\n<PROPERTIES_END>\n"
                f"<ROOTLINK_START>\n{r}<ROOTLINK_END>\n<OUTLINK_START>\n{o}<OUTLINK_END>\n<NETWORK_END>\n")
    saved = "".join([net(1, [(1, "X1")], [("SET", "Y1")]), net(2, [(1, "X2")], [("RST", "Y1")]),
                     net(3, [(3, "X0")], [("SET", "M2")]),
                     net(4, [(1, "M0")], [("RST", "M2"), ("RST", "M0"), ("CNT", "C0", "9999")]),
                     net(5, [(1, "M1")], [("TMR", "T0", "150"), ("RST", "M2"), ("CNT", "C1", "9999")]),
                     net(6, [(1, "T0")], [("TMR", "T1", "50"), ("OUT", "Y0")]),
                     net(7, [(1, "T1")], [("RST", "M1"), ("RST", "M2")])])
    text = "[DEVICE_CMT_START]\nM2::Trigger Pin For Python\n[DEVICE_CMT_END]\n<POU>\n" + saved + "</POU>"
    raw = b"\x00" * 264 + zlib.compress(text.encode("latin-1"))[2:-4]       # header + raw deflate, like ISPSoft
    nets, cm = parse(decompress(raw))
    assert [n.id for n in nets] == list(range(1, 8)) and cm == {"M2": "Trigger Pin For Python"}
    assert nets[4].outputs[0] == ("TMR", "T0", "150") and nets[2].contacts == [("RISE ", "X0")]
    st = {r[0]: r[1] for r in check(nets, 0.75, 0.25)}
    assert st == {"R1": "PRESENT", "R2": "PRESENT", "R3": "PRESENT", "R4": "PRESENT", "R5": "MISMATCH",
                  "R6": "MISMATCH", "R7": "PRESENT", "R8": "ABSENT", "R9": "ABSENT", "R10": "ABSENT",
                  "R11": "MISMATCH", "R12": "ABSENT", "R13": "ABSENT"}, st
    st = {r[0]: r[1] for r in check(nets, 15.0, 5.0)}
    assert st["R5"] == "PRESENT" and st["R6"] == "PRESENT"
    print("ok  ladder_check: header search + raw deflate, networks / timers / comments parsed, 13 requirement "
          "checks on the decoded 7-network ladder")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
        sys.exit(0)
    opt = {}
    for flag in ("--t0", "--t1", "--estop"):
        if flag in args:
            i = args.index(flag)
            opt[flag[2:]] = args[i + 1]
            del args[i:i + 2]
    path = Path(args[0]) if args else DEFAULT_ISP
    if "t0" not in opt or "t1" not in opt:
        try:
            import dataset as D
            s = D.load_settings()
            opt.setdefault("t0", s.get("plc_t0_s"))
            opt.setdefault("t1", s.get("plc_t1_s"))
            opt.setdefault("estop", s.get("estop_device") or "")
        except Exception:                                             # noqa: BLE001 - settings are optional here
            pass
    print(report(path, float(opt["t0"]) if opt.get("t0") is not None else None,
                 float(opt["t1"]) if opt.get("t1") is not None else None, str(opt.get("estop") or "")))
