# PLC Communication

The Python <-> PLC foundation: transport, device map, guarded PASS/REJECT commands, trigger handling, fault
behaviour and a commissioning window. **Everything below that was measured was measured against the ISPSoft
DVP-SS2 *simulator*. No physical PLC has been connected.**

Labels used: **VERIFIED** (observed on this machine, say where) - **USER-STATED** (given by the user, not re-derivable
here) - **INFERRED** (reasoned or from general Delta knowledge, not tested) - **UNKNOWN** - **REQUIRES PHYSICAL TEST**.
Test classes are never mixed: **FAKE PLC TEST** (in-process fake) / **SIMULATOR TEST** (ISPSoft) / **PHYSICAL PLC TEST** (none yet).

> **2026-10-05:** `final_year.isp` (saved 2026-10-04 14:26) and every backup since 2026-10-03 20:24 decode to
> the **same** 7 networks as 0.1 below (VERIFIED-FILE, `python -m plc.ladder_check`). What the ladder must do for
> this software, what it already does, and the rungs to add: [`../hardware/PLC_LADDER_REQUIREMENTS.md`](../hardware/PLC_LADDER_REQUIREMENTS.md).
> Note: the .isp header length varies between saves (0xAA in the 01:36 file, 264 bytes now); `ladder_check` searches for it.

## 0. The ladder as actually saved, and what the simulator does (2026-10-03) -- READ THIS FIRST

### 0.1 Current file: saved 2026-10-03 09:06 (VERIFIED from the file; simulator run PENDING)

The project was re-saved at 09:06 (backup `final_year_2026-10-3-9-2-10.~bak`). Decoded the same way as below:

| Net | Logic |
|---|---|
| 1 | `X1` -> `SET Y1` |
| 2 | `X2` -> `RST Y1` |
| 3 | `X0` (rising edge) -> `SET M2` |
| 4 | `M0` -> `RST M2`, `RST M0`, `CNT C0 K9999` |
| 5 | `M1` -> `TMR T0 K150`, `RST M2`, `CNT C1 K9999` |
| 6 | `T0` -> `TMR T1 K50`, `OUT Y0` |
| 7 | `T1` -> `RST M1`, `RST M2` *(new: replaces the `RST M1` that net 6 used to do)* |

What follows from the file (INFERRED from the logic; reproduced exactly by `FakeLadder` in `plc/test_simulation.py`,
**not yet observed on the simulator**):
* **The one-scan Y0 flash is fixed.** M1 now stays ON until T1 completes, so Y0 is ON for ~T1, then net 7 drops M1,
  and on the next scan T0, T1 and Y0 reset.
* **Presets are still K150 / K50 = 15 s / 5 s** at the 100 ms base, not the stated contract K15 / K5 (1.5 s / 0.5 s).
  Every REJECT therefore holds M1 for ~20 s. *Needs the ladder owner's decision; this project does not edit the ladder.*
* **REJECT acknowledge semantics.** M2 drops at once (net 5), but M1 is HELD by design. `PLCService` now treats
  "M2 cleared" as the acknowledgement for M1 (`address_map.HELD_UNTIL_DONE`, `CommandResult.cmd_held = True`). M1
  clearing later means "reject cycle finished". PASS still needs M0 *and* M2 cleared.
* **Triggers are masked during a reject cycle.** Net 5 resets M2 on every scan while M1 is ON, so a bottle reaching X0
  in that window (T0 + T1 = 2 s with the contract presets, 20 s with the saved ones) raises **no trigger**. The same is
  true if X0 rises while M2 is still ON (previous bottle not answered yet). With `watch_x0=True`, `PLCService` reads
  X0 *before* M2 in every poll and reports each such X0 edge as a `BOTTLE_UNTRIGGERED` event. `machine_cycle` records
  that bottle as FAULT "NOT INSPECTED" instead of losing it silently. This is a ladder limit on continuous handling;
  only a ladder change removes it (see 0.2).
* Only one reject can be in the PLC at a time (one T0). After a REJECT, the minimum bottle spacing is T0 + T1.

### 0.2 What a ladder change would need (proposal for the ladder owner, NOT implemented)

The software FIFO (`machine_cycle.py`) already knows when each bottle reaches the reject station. A ladder that
removes the masking would separate "inspection answered" from "fire the cylinder":
1. Make PASS and REJECT both acknowledge only the trigger (`RST M2`), not run a long timer on the command bit.
2. Let Python send the REJECT at `scheduled_reject_time - T0` (already implemented: `dispatch_at`), with T0 a short,
   fixed actuator lead time and T1 the pulse. Or keep T0 = travel and add a PLC-side shift register so several
   rejects can be in flight.
3. Never reset M2 from the reject timer rung (net 5 / net 7), so X0 edges during a reject still create triggers.

Until then, `machine_cycle` works with the ladder as it is and reports every bottle it could not handle.

### 0.3 Earlier file: saved 01:36 / 02:59 (historical; this is what the simulator measurements below used)

**Correction:** `final_year.isp` is *not* encrypted. After a 0xAA-byte header it holds a raw-deflate stream
(`zlib.decompressobj(-15)`) of ISPSoft's text project. Decoded from the file saved 2026-10-03 01:36 (**VERIFIED from the
file**; node types read as 1 = NO contact, 2 = NC contact, 3 = rising-edge contact, 13 = OUT, 15 = SET, 16 = RST):

| Net | Logic |
|---|---|
| 1 | `X1` -> `SET Y1` |
| 2 | `X2` -> `RST Y1` |
| 3 | `X0` (rising edge) -> `SET M2` |
| 4 | `M0` -> `RST M2`, `RST M0`, `CNT C0 K9999` |
| 5 | `M1` -> `TMR T0 K150`, `RST M2`, `CNT C1 K9999` |
| 6 | `T0` -> `TMR T1 K50`, `OUT Y0`, `RST M1`, `RST M2` |

(The project was re-saved 02:59 with identical logic. A second REJECT run caught Y0 ON in exactly 1 of 126 samples, at ~15.0 s, OFF again ~80 ms later, T1 still 0 -- the one-scan flash.)

Simulator behaviour matches this file exactly (**VERIFIED on the simulator**, one REJECT sampled every ~75 ms for 15 s):
M2 clears at once; M1 stays ON while T0 counts 0 -> 150 (**15.0 s**, 100 ms base); at T0 = 150 M1 and T0 clear together;
**Y0 was ON in 0 of 187 samples and T1 never left 0**; C1 +1. PASS: M0 and M2 clear within ~14-36 ms, C0 +1, Y0 never ON.

**Why Y0 never appears (follows from net 6, not from a guess):** when T0 completes, the same rung does `OUT Y0` *and*
`RST M1`. M1 is T0's enable (net 5), so on the next scan T0 drops, the rung goes false, Y0 turns OFF and T1 is reset
after one scan. Y0 is ON for about **one PLC scan**; T1 (K50 = 5 s) can never time out. No Y1/X1 interlock exists on Y0.

**Mismatch with the stated contract** (T0 ~1.5 s, T1 ~0.5 s, Y0 pulse of T1): the *earlier* backups
(`final_year_2026-10-2-*.~bak`, up to 23:31) contain exactly that design -- net 5 `(M1 or T0)... TMR T0 K15`, net 6
`(T0 or Y0) and not T1 -> TMR T1 K5, OUT Y0`. The 01:36 edit replaced it with the table above. **The ladder was not
changed by this project; this needs the ladder owner's decision.** The M0/M1/M2 meanings in section C are confirmed by the
file's own device comments; C0/C1 are counters of PASS / REJECT commands.

Software consequence (at the time): `PLCService` acknowledged a command only when the command bit *and* M2 were back
to 0, so REJECT returned `NOT_ACKED` (M1 held for 15 s). Superseded by the M1 `HELD_UNTIL_DONE` rule in 0.1.

## A. Architecture

```
 inspection engine (later)  --PLCLink-->  PLCService  --PLCClient--> Transport --> PLC ladder --> conveyor / sensor / reject
 (PASS/REJECT decision)                   plc/service.py plc/client.py  TcpTransport (sim)        owns all timing and outputs
                                          1 worker thread               SerialTransport: NOT written
```

| Piece | File | Role |
|---|---|---|
| Framing | `plc/protocol.py` | Modbus ASCII frames, LRC, exception replies. Pure functions |
| Device map | `plc/address_map.py` | The single map: addresses, meanings, safety class, write policy, command/trigger bits |
| Client | `plc/client.py` | Low-level reads, guarded writes, health. Thread-safe per transaction. Application code should not use it directly |
| **Service** | `plc/service.py` | **The one owner of the link.** Worker thread, polling, M2 trigger events, at-most-once PASS/REJECT with acknowledgement, event log |
| Transport | `TcpTransport` in `client.py` | Simulator only. The `Transport` protocol (`open/close/send/recv_frame`) is what a serial port must implement |
| Tests | `plc/test_simulation.py`, `plc/test_service.py` (fake), `plc/handshake_test.py` (simulator), `plc/commissioning.py --selftest` | see section H |
| Window | `plc/commissioning.py` | Commissioning only; not a production dashboard |

The inspection side depends on the `PLCLink` surface: `connect, disconnect, is_connected, read_bit, read_word, read_status,
wait_for_trigger, send_pass, send_reject, health_check`. No YOLO, camera or image type appears in `plc/`.
Raw `write_bit/write_word` exist only on `PLCClient` and are refused for everything except M0/M1.

Old assets: `plc file/delta_sim_test.py` is a standalone manual keypad script and **is not part of the system**. It
hard-codes M0/M1/M2/Y0/Y1 addresses (matching the current map) but labels M0 "GOOD" / M1 "REJECT"; the current contract
(M0 = PASS command, M1 = REJECT command) agrees, but that comment is not independent evidence. `plc_modbus_gui.py` does not exist.
It is superseded by `plc/`; leave it or delete it, but nothing imports it. There is **one** PLC implementation.

## B. Simulator configuration (VERIFIED 2026-10-02/03)

| Setting | Value | Evidence |
|---|---|---|
| Simulator | DVP SS2/EC3_8K (`DVPSimulator_SS2.exe`) | process list; listening on `0.0.0.0:10002` (netstat) |
| COMMGR driver | "Simulaiton SE" (sic), `CON_INTERFACE=7` | `final_year.ini`, `DriverInfo.dri` |
| Transport / address | TCP `127.0.0.1:10002` | connect succeeds |
| Protocol / station | Modbus ASCII over TCP, station 1 (the simulator answers any station: it does not enforce it) | frames exchanged; station-2 probe answered |
| ISPSoft session | ISPSoft <-> COMMGR session (port 8895) stayed connected while Python talked to port 10002 | netstat ESTABLISHED, no new COMMGR errors |

## C. Device map (single source: `plc/address_map.py`)

X and Y are octal, M/T/C decimal. Meanings were USER-STATED and are now confirmed by the decoded ladder (section 0). Addressing was **VERIFIED** by reading the simulator (and M by Delta's RUN relays
M1000..M1003 = 1,0,0,1).

| Device | Modbus addr | Type / direction | Meaning | Python access | Safety class | Sim read |
|---|---|---|---|---|---|---|
| X0 | 0x0400 FC02 | input | photoelectric sensor, bottle at station | read | input | yes |
| X1 | 0x0401 FC02 | input | start button (SET Y1) | read | input | yes |
| X2 | 0x0402 FC02 | input | stop button (RESET Y1) | read | input | yes |
| Y0 | 0x0500 FC01 | output | reject solenoid | read only | **never written** | yes |
| Y1 | 0x0501 FC01 | output | conveyor motor | read only | **never written** | yes |
| M2 | 0x0802 FC01 | internal | PLC -> Python trigger: bottle ready | read only | PLC-owned | yes |
| M0 | 0x0800 FC01/FC05 | internal | Python -> PLC PASS command; PLC resets M0 and M2 | **write 1 via PLCService** | COMMAND | yes |
| M1 | 0x0801 FC01/FC05 | internal | Python -> PLC REJECT command; PLC resets M1 and M2, then T0 -> Y0 -> T1 | **write 1 via PLCService** | COMMAND | yes |
| T0 | 0x0600 (FC01 contact, FC03 value) | timer | travel delay, intended K15 = 1.5 s (100 ms base) | read only | timer | yes |
| T1 | 0x0601 | timer | reject pulse, intended K5 = 0.5 s | read only | timer | yes |
| C0, C1 | 0x0E00, 0x0E01 (FC03 value) | counter | **UNKNOWN role** (user reports counter blocks; role not stated; the encrypted ladder cannot be read) | read only | counter | read OK, idle 0 |
| M1000 | 0x0BE8 | special relay | PLC RUN (Delta-defined) | read only | heartbeat | yes |

The previous map listed M10-M12, T3-T5, Y2, Y3 as "in the ladder list". They are not in the current contract and were
removed from the map (any valid name still resolves through `lookup()` as UNKNOWN / read-only). The older Y1 mystery
("ON but not in the list") is resolved: Y1 is the conveyor.
`.isp`/`.ini` in the working tree are newer than the ones the earlier verification used (modified 2026-10-03 01:36); the
contract above is the user's statement about that current ladder. **C0/C1 meaning and the T0/T1 timer *base* are not verifiable
from the file**; the handshake test measures T0/T1 durations and compares them with the intended values.

## D. PASS handshake (contract; Python side implemented and FAKE-tested; simulator run pending, see K)

1. X0 -> ladder sets M2. `PLCService` sees M2 rising -> `Trigger(id)`; `wait_for_trigger()` delivers it once.
2. Application inspects, calls `send_pass(trigger.id)`.
3. Service re-reads M2/M0/M1 and requires M2 = 1, M0 = 0, M1 = 0; marks the trigger answered **before** sending;
   writes M0 = 1 once.
4. PLC resets M0 and M2 -> service sees both 0 -> `ACKED` (with write->response and write->ack latency). Y0 must stay OFF.

## E. REJECT handshake

Same as D with M1, except the acknowledgement is **M2 cleared** (the 09:06 ladder holds M1 ON through T0 + T1; see 0.1,
`HELD_UNTIL_DONE`). After the ack the **PLC** runs T0 -> Y0 ON -> T1 -> Y0 OFF -> M1 OFF. Python never times or drives Y0.
The ack means "the PLC consumed the command", not "the bottle was ejected"; `machine_cycle` watches Y0 ON/OFF (status
poll, ~150 ms resolution) and records a FAULT if Y0 is never seen for a REJECT.

## F. Timing measurements

* Read latency, **SIMULATOR ONLY** (N = 200, 2026-10-03): single/block/input/word reads median ~16-17 ms, mean ~19-20 ms,
  p95 ~32 ms, p99 ~33-41 ms, max 106 ms, 0 failures; `read_status()` (now 6 transactions) mean 120 ms; connect+heartbeat+
  disconnect mean 25 ms; wrong port -> `PLCConnectionError`/FAULT after ~2.0 s (Windows retries a refused loopback connect).
  (Yesterday's run: ~12-13 ms, `read_status` 64 ms. Simulator pacing varies with machine load.)
* Write request -> response, write -> PLC ack, command -> Y0 ON, Y0 ON -> OFF: **`python -m plc.handshake_test --real`
  prints these (N, mean, median, min, max, p95, p99, failures). NOT YET MEASURED on the simulator** (see K).
* None of this is real-PLC or machine timing.

## G. Failure behaviour

| Case | Behaviour | Test class |
|---|---|---|
| wrong port / simulator down | `PLCConnectionError`, link **FAULT**, `is_connected()` False, FAULT event | FAKE + SIMULATOR |
| timeout, garbage, bad LRC, wrong station, peer close | `PLCError`, FAULT, socket closed, FAULT event, snapshot withheld | FAKE |
| command / read during FAULT or DISCONNECTED | refused before any byte is sent | FAKE + SIMULATOR |
| link lost while M2 active | trigger -> `LOST`; after explicit `connect()` M2 seen again is a **new** trigger flagged `after_reconnect` | FAKE |
| write sent, reply lost | `WRITE_FAILED`, FAULT, **not retried**, `unresolved_command` flagged | FAKE |
| written but PLC never consumes it | `NOT_ACKED`, **not retried**, flagged | FAKE |
| written, link dies before ack | `ACK_LOST`, not retried | by construction (same path), not separately tested |
| second answer to same trigger / 8 racing threads | exactly 1 write on the wire | FAKE |
| trigger not answered in `trigger_overdue_s` | `TRIGGER_OVERDUE` event; not dropped | FAKE |
| no good reply for `stale_after_s` | link state **DEGRADED** (never "connected") | FAKE |
| 20 reconnects, shutdown while connected | clean, no writes | SIMULATOR |

There is **no automatic reconnect and no automatic retry**. A reconnect never replays a command. After `WRITE_FAILED` /
`NOT_ACKED` / `ACK_LOST` the application must read the PLC and decide; the service will not decide for it.

## H. Tests (what exists and how to run)

```
python -m plc.test_simulation                       # FAKE: protocol/addressing/write policy/FAULTs (unit)
python -m plc.test_service                          # FAKE: service + FakeLadder (scan emulation of the 09:06 ladder), 13 groups
python -m plc.commissioning --selftest              # FAKE: HMI against fake PLC
python -m plc.test_simulation --real --bench 200 --faults   # SIMULATOR, read-only
python -m plc.handshake_test --fake                 # FAKE: the harness itself
python -m plc.handshake_test --real --cycles PASS,REJECT,PASS,REJECT,REJECT,PASS --wait 60   # SIMULATOR, writes M0/M1
python -m plc.handshake_test --real --sim-x0 --t0 1.5 --t1 0.5 --csv trace.csv   # ...X0 pulsed by Python, full trace
python -m line.machine_cycle                             # FAKE: the whole per-bottle cycle against FakeLadder
```

`FakeLadder` runs the seven decoded nets in order every scan, commits each scan atomically, and serves X from a
scan-latched input image, as a real PLC answers Modbus between scans. Three test flakes during development were
mid-scan reads in an earlier fake, not service bugs. It is still a FAKE: real scan time, serial timing and the real
timer base are only measured by `--real`.

## I. Simulator limitations

* The simulator answers any station and has no serial timing, no framing errors, no cable faults.
* **Trigger stimulus:** `PLCService.simulator_test_write` may pulse X0 (and X1/X2/M0/M1/M2) **only** on a loopback
  simulator link, one bit per write, logged as `SIM_TEST_WRITE`. `handshake_test --sim-x0` and the Production tab's
  "Simulate bottle (X0)" use it. On a serial (real PLC) link it is refused.
* If a second bottle arrives while M2 is still 1, or while M1 is ON, M2 shows no new edge. With `watch_x0=True` this is
  now reported (`BOTTLE_UNTRIGGERED`), not silent, but the bottle is still not inspected (0.1).
* Polling is ~20-50 ms with ~16 ms transactions: a trigger is noticed up to ~one poll period late.

## J. Physical PLC transition and COM-port ownership

**Known:** a DVP14SS2 family PLC (user: DVP14SS2-1TRW; the usual catalogue numbers are DVP14SS211R/T, so the exact variant
suffix is **UNKNOWN**), a Delta cable the user owns (type **UNKNOWN**), Modbus ASCII with LRC works against the simulator.
**INFERRED from general Delta documentation, not tested here:** SS2 CPUs have an RS-232 programming port (COM1) and an RS-485
port (COM2, terminal block), both support Modbus ASCII/RTU, factory default is ASCII 7E1 9600 baud station 1 on COM1, and the
`0x0400/0500/0600/0800` base addresses match this series. **UNKNOWN / REQUIRES PHYSICAL TEST:** COM number, baud, parity, data/stop
bits, ASCII vs RTU, station, USB-serial driver, whether the cable is RS-232 or RS-485, and whether the PLC's COM1 can be used
while a programming cable is also needed.

**Ownership (the real constraint):** a serial port is opened by one process at a time. While ISPSoft is online through COMMGR,
COMMGR holds the COM port and Python cannot open it (and vice versa). Therefore at machine runtime either (1) ISPSoft/COMMGR
is closed and Python owns the port, or (2) a second physical port is used (RS-485 COM2 via a USB-RS485 adapter for Python,
programming cable stays on COM1). Option (2) is the recommended arrangement if the ladder is still being edited; it **REQUIRES
PHYSICAL TEST**. The application must acquire the port on `connect()`, release it on `stop()` (already true for `Transport.close()`),
and tell the operator clearly when the port is busy (an open failure becomes `PLCConnectionError` -> FAULT, which is correct).
`SerialTransport` (in `plc/client.py`, pyserial) now exists: Modbus ASCII over a COM port with selectable port, baud and
format (7E1/7O1/7N2/8N1/8E1/8O1). **It is NOT TESTED ON HARDWARE** -- only its framing is tested, on pyserial's `loop://`
echo port (FAKE). The port/baud/format/station must be read from the real PLC, not assumed (Delta default 9600 7E1 station 1
is INFERRED). `PLCService.reconfigure()` switches the one service between the simulator and the serial link; the main
application's Machine tab has the selector (Simulator TCP / Real PLC serial), Connect, Disconnect and a read-only
"Test link" (10 heartbeats). Auto-connect happens only in simulator mode; a COM port is opened only on Connect. Simulator
test switches and `simulator_test_write` refuse to work on a serial link. Modbus **RTU** is not implemented.

## K. Status of the simulator write path for the 09:06 ladder: **NOT YET RUN**

The 01:36 ladder was exercised on the simulator (0.3). The 09:06 ladder has not been: on 2026-10-03 ~11:00 the
`DVPSimulator_SS2` was started standalone, but it was in STOP with no program (`M1000 = 0`), and only ISPSoft can
download the ladder. `handshake_test --real` now refuses to start in STOP and says so. Exact next steps:
1. ISPSoft: open `final_year`, set T0 = **K15** and T1 = **K5** if the 1.5 s / 0.5 s contract stands (the saved file
   has K150 / K50), compile, Simulation ON, download, RUN.
2. `python -m plc.handshake_test --real --sim-x0 --cycles PASS --csv single_pass.csv`
3. `... --cycles REJECT --csv single_reject.csv`
4. `... --cycles PASS,REJECT,PASS,REJECT,REJECT,PASS --csv sequence.csv` (the harness waits for each reject cycle to
   finish before the next X0, because a trigger during M1 would be masked). If the presets stay K150/K50, add
   `--t0 15 --t1 5` to check against them instead.
5. Freeze the contract (section C + 0.1) once those pass, then run the Production tab against the simulator
   (Simulate bottle / auto-feed).

## L. Known unknowns / next hardware verification

1. Simulator handshake result (above) and the observed T0/T1 against 1.5 s / 0.5 s.
2. C0/C1 role (and whether the ladder uses them to count bottles/rejects).
3. Physical: COM port, serial settings, ASCII vs RTU, station, cable pinout, port ownership with ISPSoft.
4. Physical: real read latency; real X0 -> M2 behaviour with debounce/bounce; M2 chatter.
5. Heartbeat/watchdog in the **ladder** (stale PC -> safe state without PC involvement). Not present in the stated contract.
6. What the ladder does if M0 and M1 are both set, or either is set with M2 = 0 (the service never does either).
