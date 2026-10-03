"""The single PLC device map: which devices exist, how they are addressed, and what is KNOWN.

Three different things are tracked separately, and must not be confused:

  1. ADDRESSING -- device name -> Modbus address/function. Verified against the running DVP-SS2
     simulator by reading it (details in `verification`).
  2. MEANING    -- what the device DOES in the machine. The ladder (`plc file/final_year/final_year.isp`)
     is a compressed text project (see SYSTEM_ROADMAP/PLC_COMMUNICATION.md section 0 for the decoded
     networks). Meanings were stated by the user and agree with the file's device comments. Python-side
     proof is the handshake test in `plc/handshake_test.py`. Anything not in that contract stays UNKNOWN.
  3. WRITE SAFETY -- whether Python may write it. X inputs and Y outputs are never writable. The ONLY
     writable devices are the two command bits M0 (PASS) and M1 (REJECT), and only the guarded
     `plc.service.PLCService` path is meant to write them.

Delta numbering: X and Y are OCTAL (X7 is followed by X10 = the 9th input); M, T, C, D are decimal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .protocol import FC_READ_COILS, FC_READ_INPUTS, FC_READ_REGS

UNKNOWN = "UNKNOWN"

# kind -> (Modbus base address, read-bit function or None, octal numbering?, bit/word capability)
# Bases follow the Delta DVP convention; each one marked VERIFIED was confirmed by reading the
# running simulator (see Device.verification / SYSTEM_ROADMAP/PLC_COMMUNICATION.md).
KINDS = {
    "X": dict(base=0x0400, bit_fc=FC_READ_INPUTS, octal=True,  words=False, direction="input"),
    "Y": dict(base=0x0500, bit_fc=FC_READ_COILS,  octal=True,  words=False, direction="output"),
    "M": dict(base=0x0800, bit_fc=FC_READ_COILS,  octal=False, words=False, direction="internal"),
    "T": dict(base=0x0600, bit_fc=FC_READ_COILS,  octal=False, words=True,  direction="timer"),
    "C": dict(base=0x0E00, bit_fc=FC_READ_COILS,  octal=False, words=True,  direction="counter"),
    "D": dict(base=0x1000, bit_fc=None,           octal=False, words=True,  direction="register"),
}
_NAME = re.compile(r"^([XYMTCD])(\d+)$")

# Writes to physical outputs and inputs are refused outright. Flipping this is a deliberate,
# reviewed change -- not something a caller can do by passing a flag.
ALLOW_OUTPUT_WRITES = False

# Current machine contract (USER-STATED, simulator-tested by the user; Python re-proves it in
# plc/handshake_test.py). If the ladder changes, change ONLY this block and re-run that test.
TRIGGER_BIT = "M2"                     # PLC -> Python: bottle at the inspection station, inspect now
PASS_BIT = "M0"                        # Python -> PLC: PASS (PLC self-clears it and M2)
REJECT_BIT = "M1"                      # Python -> PLC: REJECT (PLC self-clears it and M2, then T0 -> Y0 -> T1)
COMMAND_BITS = {"PASS": PASS_BIT, "REJECT": REJECT_BIT}
# Command bits the ladder deliberately keeps ON after consuming them: M1 enables T0 (net 5) and is reset
# only by T1 (net 7), so it stays ON for T0 + T1. For these, "consumed" = M2 cleared; the bit dropping
# later means the reject cycle finished. While M1 is ON, net 5 also resets M2 every scan, so no new
# trigger can be raised until the reject cycle ends.
HELD_UNTIL_DONE = frozenset({REJECT_BIT})
# Stated timer contract (K15 / K5 at the 100 ms base). The handshake test measures the real values.
T0_CONTRACT_S, T1_CONTRACT_S = 1.5, 0.5

# Devices Python may write: exactly the two command bits.
WRITE_ALLOWLIST: frozenset = frozenset(COMMAND_BITS.values())


# SIMULATOR-ONLY test stimulus. Never used by production code. Only these bits may ever be opened up,
# only for the duration of one write, only on the service thread, and X/Y are never among them.
SIM_TEST_BITS = frozenset({TRIGGER_BIT, PASS_BIT, REJECT_BIT, "X0", "X1", "X2"})
_sim_open = None                       # the one device a simulator test write is currently allowed to touch


class sim_test_allow:
    """Context manager: temporarily allow-list ONE bit from SIM_TEST_BITS for a simulator test write."""

    def __init__(self, device: str):
        self.device = device.strip().upper()
        if self.device not in SIM_TEST_BITS:
            raise AddressError(f"{device!r} is not a simulator-test bit (allowed: {sorted(SIM_TEST_BITS)})")

    def __enter__(self):
        global _sim_open
        _sim_open = self.device

    def __exit__(self, *exc):
        global _sim_open
        _sim_open = None


class AddressError(ValueError):
    """Not a valid device name for this PLC family."""


@dataclass(frozen=True)
class Device:
    name: str                 # e.g. "M10"
    kind: str                 # X / Y / M / T / C / D
    index: int                # numeric part, interpreted per Delta numbering (octal for X/Y)
    address: int              # Modbus address
    direction: str            # input / output / internal / timer / counter / register
    meaning: str = UNKNOWN    # what it does in the machine -- UNKNOWN unless proven
    in_ladder_list: bool = False    # named in the user's list of ladder devices (not proof of meaning)
    verification: str = ""    # how the ADDRESSING was verified
    writable: bool = False    # may Python write it (derived from the allow-list + rules)

    @property
    def read_fc(self):
        return KINDS[self.kind]["bit_fc"]


def parse(name: str) -> tuple:
    """'Y2' -> ('Y', 2). Octal rules for X/Y: 'X8' is not a device."""
    m = _NAME.match(str(name).strip().upper())
    if not m:
        raise AddressError(f"not a device name: {name!r} (expected e.g. X0, Y2, M10, T3, C0, D100)")
    kind, digits = m.group(1), m.group(2)
    if KINDS[kind]["octal"]:
        if any(c in "89" for c in digits):
            raise AddressError(f"{kind} devices are numbered in octal; {name!r} has digit 8/9")
        index = int(digits, 8)
    else:
        index = int(digits, 10)
    return kind, index


def address_of(name: str) -> int:
    kind, index = parse(name)
    addr = KINDS[kind]["base"] + index
    if not 0 <= addr <= 0xFFFF:
        raise AddressError(f"{name!r} is outside the Modbus address space")
    return addr


def is_writable(name: str) -> bool:
    """The write policy, in one place. X and Y: never (ALLOW_OUTPUT_WRITES is False). Others:
    only if on the allow-list."""
    kind, _ = parse(name)
    if _sim_open is not None and name.strip().upper() == _sim_open:      # simulator test write, see sim_test_allow
        return True
    if kind in ("X", "Y") and not ALLOW_OUTPUT_WRITES:
        return False
    return name.strip().upper() in WRITE_ALLOWLIST


def _dev(name, meaning=UNKNOWN, in_ladder=False, verification=""):
    kind, index = parse(name)
    return Device(name, kind, index, address_of(name), KINDS[kind]["direction"], meaning,
                  in_ladder, verification, is_writable(name))


_V_X = "read OK via FC02 @0x0400 on the DVP-SS2 simulator (2026-10-02); FC01 on this range returns exception 02"
_V_Y = "read OK via FC01 @0x0500; Y0 seen changing on the simulator, so it is live ladder-driven state"
_V_M = ("read OK via FC01 @0x0800+N; base confirmed by Delta's special relays: M1000..M1003 read 1,0,0,1 "
        "(RUN pattern) at 0x0BE8..")
_V_T = "contact read OK via FC01 @0x0600+N; current value read OK via FC03 @0x0600+N"
_V_C = "contact FC01 @0x0E00+N and value FC03 read OK on the simulator (idle, 0); ladder usage NOT readable"
_SRC = "USER-STATED machine contract"

# name -> (meaning, safety class, purpose/notes). Everything here is the user's stated contract;
# C0/C1 come from the decoded ladder.
CONTRACT = {
    "X0": ("Photoelectric sensor: bottle at inspection/camera station", "input, read-only",
           "its effect is M2 = 1 (inside the ladder)"),
    "X1": ("Start pushbutton: SET Y1", "input, read-only", "operator control, owned by the ladder"),
    "X2": ("Stop pushbutton: RESET Y1", "input, read-only", "operator control, owned by the ladder"),
    "Y0": ("Reject solenoid / actuator", "OUTPUT - Python must never write", "ON for T1 after T0 following M1"),
    "Y1": ("Conveyor motor", "OUTPUT - Python must never write", "set by X1, reset by X2"),
    "M2": ("PLC -> Python inspection trigger (1 = bottle ready, inspect)", "PLC-owned flag, Python reads only",
           "PLC clears it after M0 or M1"),
    "M0": ("Python -> PLC PASS command", "COMMAND - the only way to pass; PLC self-clears",
           "PLC resets M0 and M2"),
    "M1": ("Python -> PLC REJECT command", "COMMAND - the only way to reject; PLC self-clears",
           "PLC resets M1 and M2, then T0 -> Y0 -> T1 -> Y0 off"),
    "T0": ("Travel delay before the reject actuates (K15 = 1.5 s at 100 ms base)", "timer, read-only",
           "intended value from the user; the observed duration is measured in handshake_test"),
    "T1": ("Reject pulse duration (K5 = 0.5 s at 100 ms base)", "timer, read-only",
           "intended value from the user; the observed duration is measured in handshake_test"),
    "C0": ("PASS command counter (CNT C0 K9999 on the M0 rung)", "counter, read-only",
           "from the decoded ladder, net 4; counts up to 9999, the ladder never resets it"),
    "C1": ("REJECT command counter (CNT C1 K9999 on the M1 rung)", "counter, read-only",
           "from the decoded ladder, net 5; counts up to 9999, the ladder never resets it"),
}

PLC_MAP: dict = {}
for _n, (_m, _safety, _note) in CONTRACT.items():
    _k = parse(_n)[0]
    _d = _dev(_n, meaning=_m, in_ladder=True, verification={"X": _V_X, "Y": _V_Y, "M": _V_M, "T": _V_T, "C": _V_C}[_k])
    PLC_MAP[_n] = _d
SAFETY = {n: v[1] for n, v in CONTRACT.items()}
NOTES = {n: v[2] for n, v in CONTRACT.items()}

# Delta-defined special relay (meaning defined by Delta, not by this project's ladder).
PLC_MAP["M1000"] = _dev("M1000", meaning="PLC RUN monitor (Delta special relay: ON while the PLC runs)",
                        verification="read OK; M1000..M1003 = 1,0,0,1 matches Delta's RUN pattern, confirming the M base")

RUN_FLAG = "M1000"        # used by the heartbeat: a read-only, always-present, documented device


def lookup(name: str) -> Device:
    """The mapped Device, or a freshly derived one (UNKNOWN meaning, not writable) for any valid name."""
    key = name.strip().upper()
    if key in PLC_MAP:
        return PLC_MAP[key]
    return _dev(key, verification="addressing by convention only; not individually read-tested")


def group(kind: str) -> list:
    """Mapped devices of one kind, in index order."""
    return sorted((d for d in PLC_MAP.values() if d.kind == kind), key=lambda d: d.index)
