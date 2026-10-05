# What the real PLC ladder must do (and what `final_year.isp` already does)

This file is for whoever edits the Delta DVP-SS2 program in ISPSoft. It lists, requirement by requirement, what
the inspection software expects from the ladder. It says whether the current project already does it, and gives
the rungs to add where it does not.

**The ladder belongs to the user. This repository never edits it.** Everything below about the current program is
read from the file. Re-check it after every ladder edit:

```
python -m plc.ladder_check                       # reads plc file/final_year/final_year.isp + settings.json T0/T1
python -m plc.ladder_check path\to\other.isp --t0 1.5 --t1 0.5
```

Evidence labels: **VERIFIED-FILE** (decoded from the .isp on 2026-10-05) - **SIMULATOR** - **PHYSICAL** -
**PROPOSED** (not in any ladder yet).

## 1. Is the system already established in the PLC?

**Partly.** `plc file/final_year/final_year.isp` was saved 2026-10-04 14:26 (ISPSoft 3.24, DVP-SS2). It and all five
backups since 2026-10-03 20:24 contain the **same** logic (identical decoded text). VERIFIED-FILE:

| Net | Logic | Device comment in the project |
|---|---|---|
| 1 | `X1` -> `SET Y1` | X1 Start PB Green, Y1 Conveyor Motor |
| 2 | `X2` -> `RST Y1` | X2 Stop PB Red |
| 3 | `X0` (rising) -> `SET M2` | X0 PhotoElectric sensor, M2 Trigger Pin For Python |
| 4 | `M0` -> `RST M2`, `RST M0`, `CNT C0 K9999` | M0 Python Accept Flag |
| 5 | `M1` -> `TMR T0 K150`, `RST M2`, `CNT C1 K9999` | M1 Python Reject Flag, T0 Travel Done |
| 6 | `T0` -> `TMR T1 K50`, `OUT Y0` | Y0 Solenoid Valve Actuator, T1 Pulse Done |
| 7 | `T1` -> `RST M1`, `RST M2` | |

What runs on the real PLC right now is **UNKNOWN**. The first physical link (2026-10-04) was read-only, and nobody
has compared the PLC's memory with this file. Upload it from the PLC in ISPSoft (or read T0/T1 presets on the
Machine page while a REJECT runs) before trusting this table for the real machine.

## 1b. Simulation check: is the ladder correct for this software?

`plc/ladder_sim.py` runs the **real ladder from the .isp** in a scan-by-scan simulator (virtual time, no PLC) and
tests it against the software's PASS / REJECT handshake. In the app: **Production -> engineer row -> Simulation
check...** (or Machine page -> Simulation check). Command line:

```
python -m plc.ladder_sim                        # the saved ladder + settings.json
python -m plc.ladder_sim other.isp --t0 1.5 --t1 0.5 --trace
```

Each scenario is PASS / LIMIT (works, known ladder limit) / WARN (works but unsafe or unprotected) / FAIL (the
software would misbehave) / SKIP (cannot be simulated reliably). Result for the current file (2026-10-05, settings
T0 0.75 s / T1 0.25 s):

| | Scenario | Result |
|---|---|---|
| S1 | X0 -> M2, held until answered | PASS |
| S2 | PASS answer clears M0 and M2, no Y0 | PASS |
| S3 | REJECT answer clears M2 | PASS |
| S4 | Reject cycle timing equals settings | **FAIL**: Y0 starts 15 s after M1 and stays 5 s; settings say 0.75 s / 0.25 s |
| S5 | One Y0 pulse, cycle ends clean | PASS |
| S6 | Bottle during a reject cycle gets a trigger | LIMIT: no trigger for 20 s (NOT INSPECTED) |
| S7 | Second bottle before the first is answered | LIMIT: one-bottle handshake |
| S8 | Conveyor X1 latch / X2 release | PASS |
| S9 | Machine-page START/STOP (M10/M11) | WARN: ladder never reads them |
| S10 | E-stop input X3 stops the conveyor | WARN: ignored |
| S11 | Y0 blocked while the belt is stopped | WARN: no interlock |
| S12 | PC never answers | WARN: bottle passes uninspected |
| S13 | Ladder T0 fits the belt | **FAIL**: 15 s is not a travel time (the line would refuse to start) |
| S14 | Minimum spacing after a REJECT | LIMIT: 20 s |

So the ladder logic is **correct in structure** (S1-S3, S5, S8 pass, which matches the ISPSoft simulator
measurements of 2026-10-03: Y0 after 15 s for 5 s) but the **presets are wrong for this software** (S4, S13).
Changing T0 to K15 and T1 to K5 and entering 1.5 / 0.5 in the app turns S4 and S13 into PASS (checked by the
simulator's self-test on the contract ladder).

Limits of the simulator: the .isp decoder reads each network's contacts but not the series/parallel wiring.
Several contacts in one network are simulated as AND and reported (S0, and S8/S9 are SKIPped when Y1 uses OR
branches such as `X1 OR M10`). Timer base 100 ms and scan order are INFERRED from Delta behaviour and match the
simulator. It does not prove the real PLC holds this program, nor the wiring.

## 2. Requirements checklist

Status as `plc.ladder_check` reports it for the current file, with `settings.json` T0 = 0.75 s / T1 = 0.25 s:

| # | Requirement | Status (current file) | Software that depends on it |
|---|---|---|---|
| R1 | Photo-eye `X0` rising edge -> `SET M2` (one trigger per bottle) | PRESENT (net 3) | every inspection |
| R2 | `M0` (PASS) -> `RST M2` + `RST M0` in the same scan | PRESENT (net 4) | PASS acknowledge |
| R3 | `M1` (REJECT) -> `RST M2` | PRESENT (net 5) | REJECT acknowledge (`HELD_UNTIL_DONE`) |
| R4 | Reject cycle in the PLC: `M1` -> `T0` -> `Y0` for `T1` -> `RST M1` | PRESENT (nets 5-7) | the physical reject; Python never writes Y |
| R5 | `T0` preset == `settings.json plc_t0_s` | **MISMATCH**: K150 = 15 s vs 0.75 s | REJECT is sent at `trigger + travel - T0` |
| R6 | `T1` preset == `plc_t1_s` | **MISMATCH**: K50 = 5 s vs 0.25 s | pulse length; triggers masked while M1 is held |
| R7 | Conveyor: `X1` -> `SET Y1`, `X2` -> `RST Y1` | PRESENT (nets 1-2) | conveyor (PLC owns Y1) |
| R8 | `(X1 OR M10)` -> `SET Y1`, `(X2 OR M11)` -> `RST Y1` | **ABSENT** | Machine page conveyor START/STOP test buttons (`plc_operator_controls`). Today they write M10/M11, which nothing reads |
| R9 | E-stop status input (e.g. `X3`, NC contact) stops Y1 and blocks Y0, readable by the PC | **ABSENT** | `estop_device` halt + the E-stop start check |
| R10 | `Y0` only while the conveyor runs (`Y1` in series) | **ABSENT** (recommended) | machine safety |
| R11 | Triggers **not** masked during a reject cycle | **MISMATCH**: M1 held T0 + T1, its rung resets M2 every scan | continuous production; masked bottles are NOT INSPECTED |
| R12 | Answer timeout: M2 ON > N s without M0/M1 -> fail-safe | **ABSENT** | PC crash / freeze: today the bottle passes uninspected |
| R13 | PC heartbeat watchdog | **ABSENT** (software side not built either) | detect a dead PC |

## 3. What to change, in order of priority

### 3.1 Must before the first REJECT on the machine (no software change)

**Timers (R5, R6).** Set `T0` to the real lead time and `T1` to the solenoid pulse (contract: `K15` = 1.5 s,
`K5` = 0.5 s at the 100 ms base). Then enter the **same** values in Production -> engineer timing row
(`plc_t0_s`, `plc_t1_s`). How T0 is used:

```
travel  = inspection_to_reject_mm / conveyor_mm_s          (measured: Speed calibration + Line layout)
M1 sent = trigger + travel - T0          ->   Y0 = trigger + travel   (the bottle is at the cylinder)
```

T0 must be shorter than the travel time minus the inspection time. The Start check refuses otherwise.

### 3.2 Strongly recommended (no software change needed; the software already uses them)

```
Net 1   ( X1  OR  M10 )                          -> SET Y1          ; R8 start (button or PC test)
Net 2   ( X2  OR  M11  OR  NOT X3 )              -> RST Y1          ; R8 stop, R9 E-stop (X3 = NC status contact)
Net 6   T0  AND  Y1  AND  X3                     -> TMR T1 K5, OUT Y0   ; R10 interlock + R9 (no stroke when stopped)
```

Then set `settings.json` `"estop_device": "X3"` (and `"estop_active_high": false` for an NC contact). The line
halts and refuses RESET while it reads pressed.

**The hardware E-stop must cut the solenoid and motor power by itself.** X3 is only the status the software reads.

### 3.3 Fail-safe when the PC stops answering (R12) - PROPOSED

```
Net 8   M2                                       -> TMR T2 K30      ; 3 s without an answer
Net 9   T2                                       -> SET M1          ; treat as REJECT (or RST Y1 to stop the belt)
```

The time must be longer than the slowest legal answer: with a downstream camera this is camera offset + window
+ inference. The software already records such a bottle as FAULT when it finds the trigger answered by someone
else.

### 3.4 Continuous production (R11) - a design change for the ladder owner

Today M1 does two jobs: "this bottle is bad" and "run the 15 s reject cycle". While it is held, net 5 keeps M2
reset, so the next bottle cannot raise a trigger. The software also answers a REJECT only at its dispatch time,
so **M2 stays ON for about the travel time**. With this ladder the line handles **one bottle per travel time**;
any closer bottle is counted NOT INSPECTED (alarm `BOTTLE_UNTRIGGERED`).

PROPOSED protocol v2 (needs this ladder change **and** a matching software change, which is NOT implemented
yet):

| Bit | Written by | Meaning |
|---|---|---|
| M2 | PLC (X0 rising) | bottle at the camera, inspect |
| M0 | PC, right after the decision | PASS: PLC clears M0 and M2 at once |
| M1 | PC, right after the decision | REJECT accepted: PLC clears M1 and M2 at once, **no timer on M1** |
| M3 | PC, at `trigger + travel - lead` | FIRE: PLC runs `T0` (short lead) -> `Y0` for `T1`, clears M3 |

```
Net 5   M1              -> RST M2, RST M1, CNT C1 K9999
Net 6   M3              -> SET M4, RST M3                  ; M4 = reject stroke in progress
Net 7   M4              -> TMR T0 K<lead>
Net 8   T0 AND Y1 AND X3 -> TMR T1 K5, OUT Y0
Net 9   T1              -> RST M4
```

The software keeps the per-bottle FIFO (it already knows each bottle's reject time), so several bottles can be
between camera and cylinder. Two rejects must still be at least `lead + T1` apart. Software side to build:

- add M3 to `address_map.WRITE_ALLOWLIST`
- answer M1 at decision time
- send M3 from `MachineCycle._due`
- extend `FakeLadder` and the tests

Do not change one side without the other.

### 3.5 Watchdog (R13) - PROPOSED, both sides

The PC toggles `M20` every 0.5 s while the line runs. The ladder runs `TMR T3 K20` while M20 has not changed;
`T3` -> `RST Y1`. It needs M20 in the write allow-list and a toggle in `PLCService`. Neither is built.

## 4. Commissioning checks on the real PLC (record as PHYSICAL)

1. Close COMMGR (it holds COM5).
2. Machine page -> Connect. The REAL PLC badge, PLC RUN and live X/M/Y show.
3. Read `T0` / `T1` presets on the PLC (upload in ISPSoft); compare with section 1; correct the PLC and `settings.json`.
4. Break the photo-eye: X0 ON, M2 ON (trigger shown).
5. Arm, then test PASS: M0 written, M0 and M2 clear, C0 +1, no Y0.
6. Arm, then test REJECT (cylinder guarded): M2 clears, M1 held, T0 counts, Y0 for T1, M1 clears, C1 +1.
7. If R8 is added: operator START/STOP buttons move the conveyor.
8. If R9 is added: press the E-stop; the line halts, RESET is refused until it is released.
9. `python -m plc.ladder_check` after every ladder edit, and update this file's section 1.
