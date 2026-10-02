# PLC Communication

The Python <-> PLC bridge, proven against the **ISPSoft DVP-SS2 simulator**. This is a *simulator* result:
no physical Delta PLC has been connected.

> **Source of truth is the ladder in ISPSoft, and it cannot be read from this repository** (`final_year.isp` is a
> proprietary binary). So this page separates what was *measured* from what is *known about the machine*: the
> communication path and device addressing are verified; **what each device means is UNKNOWN**, and the
> write -> ladder -> read test has **not** been run because no safe command bit has been identified.

## 1. Architecture

```
 vision software --(PASS / REJECT / FAULT)--> PLCClient --Transport--> PLC (ladder) --> conveyor / sensor / reject
                                              plc/client.py            owns deterministic timing and outputs
```

| Piece | File | Role |
|---|---|---|
| Framing | `plc/protocol.py` | Modbus ASCII frames, LRC, exception replies. Pure functions |
| Device map | `plc/address_map.py` | Names -> Modbus addresses; what is verified, what is UNKNOWN; **write policy** |
| Client | `plc/client.py` | `PLCClient`: connect/disconnect/reconnect, reads, guarded writes, heartbeat, health |
| Transport | `TcpTransport` in `client.py` | **Only this changes for the real PLC.** A serial Transport is *not implemented* |
| Tests | `plc/test_simulation.py` | Unit tests (fake PLC) and the separate `--real` simulator tests |
| Commissioning window | `plc/commissioning.py` | Small test window, not a dashboard |

The application above `PLCClient` never sees sockets, frames or addresses. Python sends a high-level command and
reads state; **the PLC ladder owns actuator timing**. Python must never try to generate a millisecond pulse (the
measured link latency below is ~12 ms with outliers, which is why).

## 2. Simulator connection -- verified

| Setting | Value | Evidence |
|---|---|---|
| Simulator | **DVP SS2 / EC3_8K Simulator** (`DVPSimulator_SS2.exe`) | process list; COMMGR log "DVP SS2/EC3_8K Simulator running, Port:10002" |
| Transport | TCP | `netstat`: `DVPSimulator_SS2.exe` LISTENING on `0.0.0.0:10002` |
| Host / port | `127.0.0.1` : `10002` | `DriverInfo.dri` `LocalIPAddr="127.0.0.1" DVPSimPortNumber="10002"`; connect succeeded |
| COMMGR driver | "Simulaiton SE", ID 14114, `CommInterface=7`, timeout 3000 ms x3 retries | `DriverInfo.dri` |
| COMMGR server port | 8895 (ISPSoft <-> COMMGR) | `DriverInfo.dri` `ServerPort`; `netstat` ESTABLISHED to ISPSoft |
| Protocol | **Modbus ASCII**: `:` + hex(station, function, data) + hex(LRC) + CR LF | raw frames exchanged with the simulator |
| Station | **1** (the simulator also answers other station IDs, so it does **not** enforce the number; a real PLC will) | probe with station 2 got a reply |
| LRC | `(-sum(bytes)) & 0xFF` | simulator accepted our frames and our parser verified its replies |
| Functions | FC01 read coils, FC02 read inputs, FC03 read registers; FC05/FC06 implemented but unused | see below |
| Timeout | client default 1.0 s per transaction (configurable) | -- |

Our direct connection to port 10002 ran **alongside** the ISPSoft<->COMMGR session; that session stayed
connected and the COMMGR log showed no new errors. Whether the simulator would accept *two simultaneous
writers* was not tested.

## 3. Device map

Addressing follows Delta's convention (**X and Y are octal**; M, T, C, D decimal). `plc/address_map.py` is the single
source; its unit tests pin the arithmetic (e.g. `Y10 = 0x0508`, `M10 = 0x080A`, `M1000 = 0x0BE8`).

| Device | Modbus address | Read | Observed 2026-10-02 | Meaning | In your ladder list | Writable |
|---|---|---|---|---|---|---|
| X0 X1 X2 | 0x0400.. | **FC02** (FC01 -> exception 02) | 0 0 0 | **UNKNOWN** | yes | never |
| M0 M1 M2 | 0x0800..0802 | FC01 | 0, **1**, 0 | **UNKNOWN** | yes | no (allow-list empty) |
| M10 M11 M12 | 0x080A..080C | FC01 | 0 0 0 | **UNKNOWN** | yes | no |
| Y0 Y2 Y3 | 0x0500, 0502, 0503 | FC01 | Y0 **toggled ON/OFF while watched**, Y2 0, Y3 0 | **UNKNOWN** | yes | never |
| **Y1** | 0x0501 | FC01 | **ON** | UNKNOWN | **no -- not in the list** | never |
| T0 T1 T3 T4 T5 | 0x0600.. | contact FC01, value **FC03** | all 0 (idle) | **UNKNOWN** | yes | no |
| M1000 | 0x0BE8 | FC01 | 1 | "PLC in RUN" (Delta special relay) | -- | no |

How the addressing was verified: every device above was read successfully from the running simulator. The M base
is independently confirmed by Delta's special relays, which are defined by the PLC family and not by this ladder:
`M1000..M1003` read **1,0,0,1**, exactly the RUN pattern (M1000 = ON while running, M1001 = OFF, the two first-scan
pulses already past). The PLC is therefore in RUN. Y0 changing between reads shows the ladder is executing.

**What is NOT verified, and must stay that way until someone who has read the ladder says otherwise:**
what any of these devices *do*. The old comments in `plc file/delta_sim_test.py` (M0 = GOOD, M1 = REJECT, Y0/Y1)
are **not** confirmed by anything here and must not be reused. **Y1 being ON but absent from your device list needs
a look** -- either the list is incomplete or the output is driven by something unexpected.

From ISPSoft's compile output: the program is `final_year_conveyor_project`, 6 networks, 32 steps. The last
compile/download was 15:56 on 2026-10-02 and the saved `.isp` is byte-identical to the 15:56 `~isp`, so the
simulator is very likely running the saved ladder (not proven).

## 4. Write policy

Default: **Python cannot write anything.**
* X inputs and Y outputs are never writable (`ALLOW_OUTPUT_WRITES = False`; Y stays refused even if listed).
* M/D bits are writable only if named in `address_map.WRITE_ALLOWLIST`, which is **empty**.
* A refused write raises `PLCWriteNotAllowed` **before any bytes are sent** (unit-tested).
* The commissioning window has **no** output-forcing control; its only write control is the "command bit", disabled
  unless exactly one bit is allow-listed.

## 5. Health, faults, reconnect

* States: `DISCONNECTED` (never connected / closed on purpose), `CONNECTED`, `FAULT` (an error occurred).
* Any timeout, malformed/garbage reply, bad LRC, wrong station or closed connection raises a `PLCError` and moves to
  `FAULT`; the socket is closed because the stream may be out of step.
* In `FAULT`/`DISCONNECTED` every call raises immediately and **sends nothing** (unit-tested: the fake PLC sees no
  traffic). There are no hidden retries and no automatic reconnect -- `reconnect()` is an explicit call.
* A valid Modbus *exception* reply (e.g. illegal address) leaves the link `CONNECTED` and is counted separately.
* `heartbeat()` reads the RUN relay: read-only, returns latency and whether the PLC is in RUN, raises on any failure.
* `health()` exposes last good reply time (monotonic), last latency, last error, counters. A comms failure is never
  presented as a value and must never be read as PASS.

## 6. How to test

```
python -m plc.test_simulation                       # unit tests, fake PLC (no simulator needed)
python -m plc.commissioning --selftest              # window against a fake PLC
python -m plc.test_simulation --real                # READ-ONLY report from the real ISPSoft simulator
python -m plc.test_simulation --real --bench 400    # + read latency / reconnect / failure behaviour (read-only)
python -m plc.commissioning                         # the window against the simulator
# NOT run yet -- needs a command bit chosen by someone who has read the ladder:
python -m plc.test_simulation --real --write-test M<n> --expect Y<k>[,...]
```

ISPSoft and COMMGR must be running with the DVP-SS2 simulator started. `--write-test` allow-lists exactly the bit you
name, for that process only, always restores it afterwards, and reports PASS only if every expected device changed.

## 7. Results (2026-10-02, this machine, ISPSoft simulator)

* **Unit tests (fake PLC): pass**, 3 consecutive runs. They caught a real bug (a failed connect left the state
  `DISCONNECTED` instead of `FAULT`), now fixed.
* **Real read test: PASS.** Connected, PLC in RUN, every device in the table read back; first heartbeat ~9 ms.
* **Commissioning window against the simulator:** CONNECTED, ~9 ms latency, 96 transactions / 0 errors in a short
  poll, Y0 seen changing.

**ISPSoft simulator communication benchmark** (read-only, N = 400 per case; *not* real-PLC performance):

| Case | mean | median | p95 | p99 | min | max | failures |
|---|---|---|---|---|---|---|---|
| single bit read (FC01) | 13.0 | 14.7 | 17.1 | 18.5 | 3.2 | 24.2 ms | 0 |
| 16-bit block read (FC01) | 12.9 | 13.6 | 17.1 | 20.7 | 5.8 | 97.5 ms | 0 |
| input read (FC02) | 12.9 | 12.8 | 17.2 | 24.5 | 4.8 | 64.3 ms | 0 |
| word read (FC03, T0..T5) | 12.3 | 12.3 | 16.9 | 17.8 | 3.7 | 22.9 ms | 0 |
| `read_status()` (5 transactions) | 64.1 | 63.8 | 81.0 | 87.6 | 46.0 | 192.9 ms | 0 |
| connect + heartbeat + disconnect (N = 50) | 21.5 | 18.9 | 39.8 | 42.7 | 4.9 | 42.7 ms | 0 |

Latency is ~12 ms regardless of block size, so cost is per transaction: batch reads (`read_many`,
`read_words_many`). A wrong port ends as `PLCConnectionError` -> `FAULT` after ~2 s (Windows retries a refused
loopback connect). **Write latency, write -> response latency and repeated-command reliability were not measured:
they need a write, and no command bit has been chosen.**

## 8. Limitations -- simulator vs the real PLC

* This is a **simulator**; its pacing (~12 ms per reply), station handling and single local TCP link say nothing
  about a wired DVP. A real PLC will enforce the station number and add serial timing.
* The real link is probably serial (RS-485) or a Delta Ethernet module -- **not known**: PLC model, port, baud rate,
  parity/data bits and ASCII vs RTU are all unknown here. A serial `Transport` does not exist yet.
* Device **meanings**, the command bit, the safe test point and whether the saved `.isp` matches what the simulator
  runs are unverified. Y1's behaviour is unexplained.
* `time.monotonic()` freshness / Windows scheduling jitter apply; Python is not a real-time controller.

## 9. Migration to the real Delta PLC (plan, not done)

1. Read the real PLC's COM settings (ISPSoft hardware config) and wiring; record model and station.
2. Write a `SerialTransport` (pyserial; ASCII or RTU per the PLC) implementing the same four `Transport` methods.
3. Re-run `python -m plc.test_simulation --real`-style read-only checks against the PLC with **outputs de-energised
   or the reject disabled**, before any write.
4. Re-measure latency on the real link; budget it in the timing model ([HARDWARE_INTEGRATION.md](HARDWARE_INTEGRATION.md)).
5. Only then allow-list a command bit, and add a heartbeat/watchdog the **ladder** enforces (a stale PC must put the
   machine in a safe state without PC involvement).

## 10. Decision needed to finish this phase

The write -> ladder -> read test needs **one internal M bit that the current ladder treats as a software command**
(and what it should change). That choice must come from the ladder, not from guessing. See the final report.
