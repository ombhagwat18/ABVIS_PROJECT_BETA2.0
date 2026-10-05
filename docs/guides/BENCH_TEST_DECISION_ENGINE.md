# Bench test: PLC -> vision -> decision engine -> PLC

How to check that the vision system decides PASS / REJECT on its own and the PLC acts on it, at home or on
the bench, without running the full line. Written 2026-10-04 while connecting the real DVP-SS2.

## 1. What is connected to the decision engine

Only the **Production** tab drives the decision engine. The other tabs only look or test.

| Tab | Talks to PLC? | Uses the decision engine (`decision.py`)? | Purpose |
|---|---|---|---|
| **Production** | Yes: reads M2 (+X0), writes M0/M1 | **Yes** (`machine_cycle.Inspector` -> `decision.decide()`) | The real automatic line |
| Machine | Yes: reads everything; operator-armed PASS/REJECT only | No (the operator is the "decision") | Commissioning, I/O view, manual handshake test |
| Live | No | No (classifier/YOLO verdict on screen only) | Watching what the models see |
| Camera | No | No | Measuring camera settings (sharpness, FPS) |

The automatic path:

```
bottle in front of the photo-eye (X0)
  -> ladder net 3: X0 rising edge SETs M2
  -> PLCService sees M2 rise -> Trigger #n
  -> Inspector: frames captured strictly AFTER the trigger, every line camera
  -> classifier (+ YOLO) -> decision.decide(): frame vote, camera fusion FAULT > REJECT > PASS
  -> PASS  : M0 now                          -> ladder: C0 +1, M2 cleared
  -> REJECT: M1 at trigger + travel - T0     -> ladder: T0 -> Y0 (solenoid) for T1, C1 +1
  -> FAULT : rejected by default (fault_action = "REJECT")
  -> daily production CSV
```

## 2. Simulator switches vs the real PLC

The Machine tab's **SIMULATOR TEST** switches (X0 X1 X2 M2 M0 M1, Pulse X0) are enabled **only** for the ISPSoft
simulator (TCP 127.0.0.1). On the real PLC they stay grey. This is intentional:

- **X inputs cannot be forced over Modbus on a real PLC.** Every scan the PLC reloads X0..X17 from the input
  terminals, so a written X0 would be overwritten within one scan. A software X0 switch would not work.
- **Writing M0/M1/M2 directly on the real machine bypasses the handshake rules** (one answer per trigger,
  preconditions re-read, never retried) and can fire the reject cylinder. The code allows only M0/M1, and only
  through the guarded PASS/REJECT path.
- For the real PLC there is a separate **OPERATOR TEST** row (added 2026-10-04 on request, for commissioning and
  testing; remove it later by setting `"plc_operator_controls": false` or deleting the row):

  | Button | Writes | Needs in the ladder |
  |---|---|---|
  | Conveyor START | M10 pulse (1, 0.3 s, 0) | an M10 contact in **parallel with X1** in net 1 (`SET Y1`) |
  | Conveyor STOP | M11 pulse | an M11 contact in **parallel with X2** in net 2 (`RST Y1`) |
  | Virtual bottle | M2 <- 1 | nothing (the ladder already clears M2 on M0/M1) |

  Enabled only when `settings.json` `"plc_operator_controls": true`, the link is CONNECTED, the PLC is in RUN and
  the row's **arm** box is ticked; arm clears after every press (one press = one write). Virtual bottle is refused
  while M2, M0 or M1 is ON. Y0/Y1 and X are still never written. Software STOP is **not** an emergency stop.
  Code: `PLCService.operator_write` (`plc/service.py`), `address_map.OPERATOR_BITS`; test:
  `test_operator_write_is_opt_in_and_narrow` in `plc/test_service.py`.

## 3. How to bench test at home with the real PLC (no code change)

Use the real input, not a software switch: the trigger is then exactly the one the line will see.

1. **Free the COM port**: in COMMGR, *Delete Driver*, then exit COMMGR (check the tray); go offline in ISPSoft.
   One program per COM port. The PLC keeps running its program without ISPSoft.
2. **Link**: Machine tab -> Real PLC (serial) -> `COM5` (Silicon Labs CP210x), 9600, 7E1, station 1 -> Connect.
   Header must show **PLC ON**; X0/M2/Y0 show live ON/OFF; PLC must be in **RUN**.
3. **Trigger source** (pick one):
   - the photo-eye itself: pass a bottle or your hand through the beam; or
   - a push button / toggle wired in parallel to X0 (same 24 V common as the sensor) = a bench "bottle" switch.
   Check in the Machine tab: X0 ON -> M2 ON.
4. **Manual handshake first** (Machine tab): trigger -> tick **arm** -> PASS (M0). Expect `ACKED`, M2 off, C0 +1.
   Repeat with REJECT (M1): M1 stays ON through T0, then Y0 pulses for T1, C1 +1.
   **With the cylinder connected and air on, REJECT moves the cylinder.** Turn the air off for a dry test.
5. **Automatic (decision engine)**: Production tab -> pick the camera(s) -> Start line. Put a good bottle in view
   and trigger: expect PASS -> M0. Put a defective bottle (or cover the camera) and trigger: expect REJECT/FAULT
   -> M1 -> Y0. Every bottle appears once in the Production table and the daily CSV.
6. **Safety checks**: Production STOP button latches (no M0/M1 sent while halted); 3 FAULT bottles in a row latch
   (`fault_latch_after`). Hardware E-stop input is **not configured yet** (`estop_device` unset).

## 4. Settings that must be right before step 5

| Setting (`settings.json`) | Now | Needed |
|---|---|---|
| `plc_mode` / `plc_com` | serial / COM5 | as is (matches COMMGR: RS-232 ASCII 9600 7E1 station 1) |
| `plc_t0_s`, `plc_t1_s` | 0.75 / 0.25 | **must equal the T0/T1 presets in the PLC**. Saved ladder has T0 K150 = 15 s, T1 K50 = 5 s |
| `inspection_to_reject_mm`, `conveyor_mm_s` | 0 / 0 (not measured) | measure; conveyor ~1524 mm in 14-16 s = ~95-110 mm/s |
| `line_cameras` | [2, 3] | indices of the EMEET cameras actually enumerated |
| `camera_controls` | not set (autofocus/auto-exposure ON) | lock focus/exposure once mounted, to stop flicker |
| `estop_device` | not set | a spare X input wired to the E-stop NC contact |

## 5. Evidence labels

Everything above the PLC link was run as software self-tests with fakes (`python machine_cycle.py`,
`python -m plc.test_service`). The real-PLC link on COM5 was proven by COMMGR/ISPSoft (2026-10-04), **not yet by
this software**: record the first successful Machine-tab connection and handshake as PHYSICAL evidence in
`docs/roadmap/PLC_COMMUNICATION.md`.
