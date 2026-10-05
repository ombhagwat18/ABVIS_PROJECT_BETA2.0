# Production HMI and physical commissioning

This guide covers how the line is operated from the one desktop app (`python gui.py`), and the order in which
the physical machine should be commissioned. Software status: **SOFTWARE VERIFIED** (self-tests with fakes).
**Not one step below has been completed on the physical machine yet.** Record each result in
`docs/roadmap/FEATURE_STATUS.md` with its evidence label (PHYSICAL) when it is done.

## 1. Two modes

| Mode | Rail shows | Who |
|---|---|---|
| **OPERATOR** (default at first start) | Production, History, Health | Shift operator |
| **ENGINEER** (header button; optional PIN in Settings) | Every page, plus the ENGINEER row on Production | Commissioning, maintenance, ML |

The mode is remembered in `settings.json` `ui_mode`. Nothing is removed in operator mode, only hidden.

## 2. Machine states (one source: `machine_state.py`)

`OFFLINE -> NOT_READY -> READY -> INITIALIZING -> RUNNING <-> INSPECTING -> STOPPING`, plus `FAULT`, `E_STOP`
and `COMMUNICATION_FAULT`. The large banner on Production and the LINE lamp in the status bar always show
the same state.

START INSPECTION is enabled only in READY. That requires every start check to pass:

- PLC connected
- PLC in RUN
- cameras selected
- models present
- timing valid (T0, travel, camera stations, saved speed calibration)
- E-stop released
- recipe valid

## 3. Operator procedure

1. Engineer has connected the PLC (Machine page, **Connect**). The real PLC is never connected automatically.
2. Production: type **Job** and **Product**. Both are stored with every bottle.
3. Read the START CHECKS list until the banner says **READY**.
4. **START INSPECTION**. The banner shows INITIALIZING while models load and cameras deliver a first frame, then
   RUNNING. If a camera gives no image within 8 s, the line does not start and CAMERA_DISCONNECTED is raised.
5. During production you see:
   - both cameras live, with fps and frame age
   - the current bottle: id, result, defect, confidence, per-camera result, measured stage times, PLC status
   - the counters: TOTAL, PASS, REJECT, FAULT, NOT INSPECTED, MISSED REJECT, QUEUE
   - recent bottles, with active alarms showing message and action
6. **STOP** stops answering new triggers and stops the cameras. Bottles that were still being inspected end as
   FAULT, and their plc_status says "remove by hand".
7. **RESET FAULT** clears a software HALT (refused while the hardware E-stop input reads pressed) and
   acknowledges alarms.

The software HALT, alarms and STOP are **not** a safety function. The hardware E-stop must remove actuator
power by itself.

## 4. Tests without production

| Button | What it does | PLC touched? |
|---|---|---|
| CAMERA TEST | Chosen line cameras live, with brightness / contrast / sharpness / clipping. "Capture test frame" saves PNGs and quality.json to `captures/camtest_*` | No |
| TEST INSPECTION (no PLC) | Bottle in front of the cameras: same models, decision engine and recipe as production. Shows the result, reason, AI time and overlay; files in `captures/TEST_*` | No |
| Machine page: test PASS / REJECT | Answers one pending trigger; requires the arm checkbox, one write per arm | Yes (M0 / M1) |
| Machine page: operator test | Conveyor M10 / M11 pulses, virtual bottle M2; only with `plc_operator_controls: true` | Yes |
| Production: Simulate bottle (engineer) | X0 pulse, **simulator only** (refused on a serial link) | Simulator |

## 5. Engineer: timing without an encoder

There is no encoder. Every bottle position is a time, `trigger + distance / speed`, and the speed must be
**measured**.

1. **Speed calibration...** (engineer row):
   1. Put two marks on the belt and measure the distance between them.
   2. Run the conveyor.
   3. Press *Bottle at mark A* and then *Bottle at mark B*, or type stopwatch times. Do this 3 or more times.
   4. Press *Calculate*. Spread above the tolerance (default 10 %) is refused: check belt slip and load.
   5. Press *Save speed*. Speed, tolerance and the timestamped record go to `settings.json`.
2. **Line layout / camera stations...**:
   - Photo-eye -> reject cylinder distance (mm). The photo-eye is Camera 1's position (the trigger point).
   - For each line camera: name, role, **offset mm downstream of the photo-eye** (0 = at it), wall side,
     optional `station_x` (where the inspected bottle sits in that camera's frame, 0-1) and "judges parts"
     (e.g. `cap` for a cap/neck camera, so it is not blamed for a label it cannot see).
   - *Check* lists every problem:
     - downstream camera without a measured speed
     - camera at or after the reject station
     - decision ready after the REJECT dispatch time plus margin
3. Timing row: T0 / T1 **must equal the ladder's presets in the PLC** (read T0 on the Machine page). Late limit,
   frames per camera, window, capture delay and FAULT -> REJECT/PASS are set there too.

### Sequential cameras (opposite walls, Camera 2 shifted toward the exit)

Camera 2's frames are taken at `trigger + offset_2 / speed`, inside its own window. They are bounded by the next
bottle's window, and judged by its role.

If another bottle was sensed (triggered or not) closer than `window + 2 x uncertainty`, Camera 2's evidence is
refused as **CAMERA_ASSOCIATION_FAULT** and the bottle becomes FAULT. The rule is: never the wrong bottle.

**With the current ladder** M2 stays ON until the bottle is answered, so a downstream camera adds its travel time
to the minimum bottle gap. Bottles that arrive sooner are NOT INSPECTED (alarm BOTTLE_UNTRIGGERED).

### Recipe (what a complete product is)

Engineer row -> **Recipe...**: the anchor class (bottle) and the parts that must belong to it (cap, label), where
to search (fractions of the bottle height from its top) and where each must sit (zone). It is checked against the
detector's classes and refused while the line runs. Every run records the recipe hash.

### Cameras that drop out

While the line runs, a dead line camera is reopened in the background every 3 s. Bottles inspected in the gap are
FAULT (camera fault), and the pane shows the number of reconnects. Every attempt is in `logs/camera.log`.

## 6. Engineer: models (never automatic)

Training (Train page, `train.py`, `model_bench.py`) only creates **CANDIDATES**. Models page:

1. **Validate...** needs a held-out test result on record and a written real-camera validation, e.g.
   "EMEET cam 2, 10 good + 10 no-cap bottles, 20/20 correct".
2. **Approve**.
3. **ACTIVATE**, with the line stopped. The previous model is ARCHIVED and the deployment is logged in
   `models/deployments.jsonl`.
4. **Roll back** re-activates the previous deployment of that kind.

Detector activation checks the weights' sha256 against the candidate's training record.

`missing_cap` is a critical class (`settings.json` `critical_classes`). Every classifier currently shows
*NO test positives*: no classifier can detect a missing cap until missing-cap images are labelled and trained.

## 7. Traceability

Each START creates a run in `projects/<slug>/production/production.db`. A run records:

- job and product
- recipe hash and body
- model ids
- SIMULATOR or REAL PLC
- line settings

Every bottle gets exactly one row with these fields:

- run_id, inspection_id
- final result, AI decision, defects, confidence
- command and PLC outcome
- timings
- per-camera window and frame times
- evidence path

Evidence images follow `evidence_policy`: NONE / ALL / REJECT_ONLY / FAULT_ONLY / REJECT_AND_FAULT (the default)
/ SAMPLE:n. Alarms are stored with raise, acknowledge and clear times. The daily CSV
(`production/<date>.csv`) is still written. The History page reads the database.

### Logs and shifts

`logs/` holds one file per subsystem (app, camera, ai, plc, machine, alarm, production), one line per event. Search
them on the Health page. History reports per day or per shift (`settings.json` `"shifts"`, default A 06-14 /
B 14-22 / C 22-06; the night shift runs across midnight).

## 8. Physical commissioning order (record each as PHYSICAL when done)

| # | Step | Pass criterion | Where |
|---|---|---|---|
| 0 | Make the ladder meet `docs/hardware/PLC_LADDER_REQUIREMENTS.md` (at least T0/T1; ideally M10/M11, X3 E-stop, Y0 interlock); `python -m plc.ladder_check` | R1-R10 PRESENT | ISPSoft |
| 1 | Close COMMGR, connect the real PLC read-only | Machine: CONNECTED, REAL PLC badge, PLC RUN, X/Y/M/T/C live | Machine |
| 2 | Read T0 / T1 presets in the PLC | Values written down; `plc_t0_s` / `plc_t1_s` set to them | Machine, timing row |
| 3 | Photo-eye | X0 toggles with a bottle; M2 rises (trigger shown) | Machine |
| 4 | Cameras on separate USB 3 ports | Both stream the requested resolution together (no bandwidth note) | CAMERA TEST |
| 5 | Enclosure lighting (matte interior, ~6500 K, no LED in the opposite lens) | Brightness steady, clipped < 1 %, sharpness stable; capture test frames | CAMERA TEST |
| 6 | Camera controls locked (focus / exposure / WB / rotation) | Read back in Camera page; same values every open | Camera |
| 7 | Speed calibration | >= 3 runs, spread under tolerance, saved | Speed calibration |
| 8 | Line layout | Distances measured with a tape; Check = OK | Line layout |
| 9 | AI on the machine | TEST INSPECTION on known GOOD, missing-cap, missing-label bottles: correct result, recorded | TEST INSPECTION |
| 10 | PASS command | Arm, then test PASS on a real trigger: M0 ACKED, no Y0 | Machine |
| 11 | REJECT command (cylinder guarded, people clear) | Arm, then test REJECT: M1 ACKED, Y0 pulse after T0, bottle ejected at the cylinder | Machine |
| 12 | Line, one bottle at a time | GOOD -> PASS (M0), defect -> REJECT (Y0 hits the bottle); rows in History | Production |
| 13 | Sequential cameras | Same inspection_id carries both cameras; no association fault at normal spacing | Production / History |
| 14 | Multi-bottle | FIFO order kept; NOT INSPECTED counted (not lost) when the spacing is too short | Production |
| 15 | Latency | History / CSV timings: trigger->decision and trigger->command p50 / p95 / max; Y0 error vs schedule | History export |
| 16 | E-stop | Hardware cuts actuator power; with `estop_device` set, the line halts and RESET is refused while pressed | Production |
