# Hardware Integration

The physical system the software will eventually be connected to. **Nothing on this page has been verified
against the real machine from within this repository.** The hardware list below is what the project owner
describes; the repository itself contains no measurements, drawings or test logs for it.

## Hardware context (as stated by the project owner)

| Item | Role |
|---|---|
| Conveyor | Moves 250 ml Bisleri bottles past the inspection point to the reject location |
| 2 x EMEET NOVA 4K cameras | Image acquisition. Both enumerate and stream on the dev PC (see Camera notes: one 1080p stream per USB 2.0 hub); not yet mounted on the machine |
| Photoelectric bottle sensor | Detects a bottle arriving (not represented in software) |
| Delta DVP-series PLC, programmed in ISPSoft | Deterministic machine control |
| Festo DSNU cylinder + 5/2 solenoid valve | Reject actuator |
| Controlled LED lighting | Consistent illumination (not represented in software) |
| Inspection enclosure | Controlled imaging environment |

## What exists in the repository for this hardware

| Item | State |
|---|---|
| `plc file/delta_sim_test.py` | Standalone manual sender (Modbus ASCII over TCP to `127.0.0.1:10002`, station 1) for the ISPSoft simulator. Not part of the application |
| `plc file/final_year/` | An ISPSoft project (`.isp`, `.ini`). The `.isp` is a proprietary binary; its ladder logic **cannot be read here**. The `.ini` shows it is set to the ISPSoft *simulation* driver |
| Camera code (`infer.py`) | Index probing, DirectShow, driver-default capture settings |
| Timing / sensor / actuator code | **None** |

> ### PLC addresses and I/O mapping are NOT yet verified.
>
> The simulator script assumes a few internal-relay and output addresses, but there is no evidence in the
> repository that the ladder program uses them, and the PLC model, wiring, input assignments and output
> assignments are not documented. **Do not treat any address in the scripts as correct.** To verify, export the
> ladder (PDF/screenshots), the device comment table and the hardware configuration from ISPSoft, and compare.
> This document deliberately lists no addresses.

## Software/PLC boundary (conceptual -- design intent, not implemented)

A result handshake will be needed so that a late, missing or unacknowledged result leads to a safe outcome
chosen in the PLC. The elements to design and verify once the PLC is accessible: a result signal from software,
an acknowledgement, a heartbeat/watchdog in each direction, a result timeout, a fault/reset path, and a
counter strategy. None exists in code yet. See [INDUSTRIAL_ARCHITECTURE.md](../roadmap/INDUSTRIAL_ARCHITECTURE.md) for the
responsibility split (software decides *what*; the PLC decides *how the machine responds*).

## Timing: the equation that decides whether this works

A bottle that is rejected must still be at the reject actuator when the actuator fires.

```
available travel time  =  (distance from sensor/camera to reject position)  /  (conveyor speed)
```

The whole software + machine chain must complete inside that time, with margin:

```
capture
+ preprocessing
+ inference
+ decision
+ communication (PC -> PLC)
+ PLC scan / response
+ actuator response (valve switching + cylinder travel)
        <  available travel time
```

### What is known and what is not

| Quantity | Value in the repository |
|---|---|
| Sensor/camera-to-reject distance | **Unknown -- not recorded** |
| Conveyor speed | **Unknown -- not recorded** |
| Capture latency | **Unmeasured.** Frame timestamps are taken when `read()` returns, not at exposure |
| Preprocessing + inference | Measured only on the development laptop: YOLOv8n ~19 ms model-only / ~29 ms end to end on an RTX 3050 Laptop GPU (batch 1, under memory pressure). Not measured on the target machine or with real frames |
| Decision | Not implemented, so not measurable |
| PC -> PLC communication | **Unmeasured** |
| PLC scan / response | **Unmeasured** |
| Valve + cylinder response | **Unmeasured** (a datasheet value is not a measurement of this installation) |

**Actual hardware measurements are required** before any claim that the chain fits. The timing model is Phase 5
of [PROGRESS_PLAN.md](../roadmap/PROGRESS_PLAN.md), and nothing in the software should hard-code a value for
these quantities until they are measured.

## Camera notes

- The Live tab captures at driver defaults. The Production line requests `line_capture_wh` (default 1920x1080) with
  `line_fourcc` (MJPG) and shows what the camera actually delivers.
- No exposure, gain, focus or white-balance control; no automatic reconnect (a dead camera stays FAULT until
  restarted); no hardware trigger.
- On Windows `time.monotonic()` has ~15.6 ms resolution; frame sequence numbers give the strict ordering.
- **EMEET Nova 4K, measured on the development PC 2026-10-03 (cameras on the desk, NOT mounted on the machine):**
  DirectShow order is 0 = USB2.0 HD UVC WebCam (laptop), 1 = Iriun Webcam (virtual), **2 and 3 = EMEET SmartCam Nova 4K**
  (`infer.camera_names()` reads this; the Production tab picks the EMEETs by default). One EMEET alone: 640x480 default,
  1920x1080 MJPG 30 fps, 3840x2160 MJPG ~14 fps, ~1 s to open. Both at once: **only one stream at 1920x1080 or
  1280x720** (the second returns no frames, either open order, DSHOW or MSMF). Both stream at 640x480 (~30 fps each).
  Cause: both are on the same Generic USB 2.0 hub (`VID_1A40&PID_0101`, Hub #3, ports 2 and 3); UVC isochronous bandwidth
  is reserved per stream. **Fix: plug each EMEET into its own PC USB port (different root hub, ideally USB 3).**
  Until then set `line_capture_wh` to `[640, 480]` for a two-camera line.
- Placement, height, angle, field of view, lighting, reflection, focus and exposure in the current enclosure: **not
  started** -- needs the cameras mounted on the machine. Start with one camera (1920x1080), check bottle/cap/label
  visibility in the Production tab's live view and the YOLO boxes on the evidence image, then add the second.

## First hardware checklist (for Phase 5-8)

1. Measure distances and conveyor speed.
2. Export the ISPSoft ladder + I/O list; verify PLC model and protocol.
3. Measure sensor-to-capture and capture latency with the real camera.
4. Measure PC -> PLC round trip and actuator response.
5. Compute the budget; only then build reject timing.
6. Validate with known good and deliberately defective bottles, including failure injection.
