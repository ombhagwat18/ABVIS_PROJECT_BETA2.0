"""The single PLC device map: which devices exist, how they are addressed, and what is KNOWN.

Three different things are tracked separately, and must not be confused:

  1. ADDRESSING -- device name -> Modbus address/function. Verified against the running DVP-SS2
     simulator by reading it (details in `verification`).
  2. MEANING    -- what the device DOES in the machine (GOOD, REJECT, CONVEYOR, ...). The ladder
     (`plc file/final_year/final_year.isp`) is a proprietary binary that cannot be read from here,
     so every meaning is UNKNOWN unless something other than a guess proves it. Do not fill one in
     from habit or from the old `delta_sim_test.py` comments.
  3. WRITE SAFETY -- whether Python may write it. Default: nothing. X inputs and Y outputs are never
     writable here; an M/D write needs an explicit allow-list, chosen by someone who has read the ladder.

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

# Writes to physical outputs and inputs are refused outright in this phase. Flipping this is a
# deliberate, reviewed change -- not something a caller can do by passing a flag.
ALLOW_OUTPUT_WRITES = False

# Devices Python may write. EMPTY on purpose: choosing a command bit needs the ladder.
WRITE_ALLOWLIST: frozenset = frozenset()


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
    if kind in ("X", "Y") and not ALLOW_OUTPUT_WRITES:
        return False
    return name.strip().upper() in WRITE_ALLOWLIST


def _dev(name, meaning=UNKNOWN, in_ladder=False, verification=""):
    kind, index = parse(name)
    return Device(name, kind, index, address_of(name), KINDS[kind]["direction"], meaning,
                  in_ladder, verification, is_writable(name))


_V_X = "read OK via FC02 @0x0400 on the DVP-SS2 simulator (2026-10-02); FC01 on this range returns exception 02"
_V_Y = "read OK via FC01 @0x0500; value changed between reads, so it is live ladder-driven state"
_V_M = ("read OK via FC01 @0x0800+N; base confirmed by Delta's special relays: M1000..M1003 read 1,0,0,1 "
        "(RUN pattern) at 0x0BE8..")
_V_T = "contact read OK via FC01 @0x0600+N; current value read OK via FC03 @0x0600+N (all 0 when idle)"

# Devices the user listed as appearing in the current ladder. Presence in the list is NOT a meaning.
_LADDER_LIST = ["X0", "X1", "X2", "M0", "M1", "M2", "M10", "M11", "M12",
                "T0", "T1", "T3", "T4", "T5", "Y0", "Y2", "Y3"]

PLC_MAP: dict = {}
for _n in _LADDER_LIST:
    _k = parse(_n)[0]
    PLC_MAP[_n] = _dev(_n, in_ladder=True, verification={"X": _V_X, "Y": _V_Y, "M": _V_M, "T": _V_T}[_k])

# Observed live but NOT in the user's list -- kept so nothing is hidden from the operator.
PLC_MAP["Y1"] = _dev("Y1", in_ladder=False,
                     verification=_V_Y + "; observed ON while not in the user's ladder-device list: investigate")

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
