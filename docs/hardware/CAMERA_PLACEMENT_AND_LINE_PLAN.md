# Camera placement, lighting and line timing plan

Evidence labels as elsewhere in this repo: **USER-STATED** (from the project owner), **COMPUTED** (arithmetic from
user-stated numbers), **ASSUMED** (a typical value that must be measured on the machine), **SOFTWARE** (what the code does today).
Nothing here has been verified on the physical machine.

## 1. Known geometry (USER-STATED)

| Item | Value |
|---|---|
| Conveyor length / transit time | 5 ft (~1524 mm) in 14-16 s |
| Belt speed (COMPUTED) | **~95-110 mm/s** (roadmap uses ~90 mm/s from 1460 mm / 15-17 s). Measure it with a stopwatch + marked bottle |
| Enclosure (L x W x H) | 450 x 320 x 400 mm |
| Lighting | 2 x Wipro LED strips, 6500 K cool white; black matte sheet as background |
| Cameras | 2 x EMEET Nova 4K, one on each side of the belt |
| Bottle (ASSUMED, measure it) | 250 ml PET: ~60 mm diameter, ~150-170 mm tall |

At ~100 mm/s the bottle needs ~4.5 s to cross the 450 mm enclosure and moves ~60 mm during the 0.6 s inspection window.
That is slow: there is no need for a hardware trigger or a global shutter, but exposure must still be short and fixed.

## 2. Do the two cameras "create noise" for each other?

No. Both are passive; they do not interfere electrically or optically. The real problems are:

1. **Seeing each other.** Camera B appears in camera A's frame (and its lens reflects the LED strip). On a black background a
   small black body is mostly invisible, but a lens glint is not.
2. **LED glare.** PET is glossy; a strip facing the camera gives a vertical white streak that hides the meniscus (`water_level`)
   and label print (`damaged_label`).
3. **USB bandwidth.** Measured 2026-10-03: two EMEETs on the same USB 2.0 hub deliver only one 1080p stream. Each camera needs its
   **own PC USB port on a different root hub** (ideally USB 3). Until then use 640x480 for both. This is the biggest practical risk.

## 3. Recommended placement

```
            TOP VIEW  (belt runs left -> right, enclosure 450 x 320 mm)

   wall A  +--------------------------------------------------+
           |   [LED strip A]                                  |
           |        CAM A  ->                                 |
           |                  .                               |
   belt -->|=====================( bottle )====================|--> to reject
           |                               .                  |
           |                         <- CAM B                 |
           |                                  [LED strip B]   |
   wall B  +--------------------------------------------------+
                 |<- stagger ~120 mm ->|
```

- **Opposite walls, staggered along the belt by about 120 mm** (so neither camera is in the other's field of view; verify with the
  FOV measurement in section 7), both aimed at the bottle's centre line. Not face to face on one axis.
- **Height:** optical axis at the bottle's mid-height (~80-85 mm above the belt), camera level (no tilt) so the cap and the base both
  stay inside the frame and the meniscus is not foreshortened.
- **Distance:** ~150-200 mm from the bottle centre line (ASSUMED). The 320 mm width leaves room: belt centred => ~160 mm to each wall.
- **Portrait orientation:** the bottle is tall and narrow and the model input is 192x448 (tall). Mount each camera rotated 90 degrees
  and set `"rotate": 90` in `settings.json` `camera_controls`. In portrait the long (vertical) field of view is the wide one.
  *Check:* needs vertical coverage >= bottle height + 20% margin (~200 mm at 160 mm distance => ~65 degrees vertical FOV; the Nova 4K's
  real FOV must be **measured**, not assumed).
- **Coverage honesty:** one camera at ~160 mm sees a ~160-degree arc of a 60 mm cylinder, but the outer ~40 degrees on each side is
  grazing and foreshortened. Two opposite cameras therefore give roughly **240 degrees of good-quality coverage, not a true 360**.
  If the label seam or a defect can sit at the left/right edge, add a third camera at 90 degrees or accept the blind sides.
- **Each camera sees "its half"**, as you described: A inspects the front half, B the back half. The decision engine already fuses
  cameras FAULT > REJECT > PASS (`decision.py`), so a defect seen by either camera rejects the bottle.

## 4. Lighting

- **Mount both strips above and slightly behind each camera's plane, angled down at ~30-45 degrees, with a diffuser** (opal acrylic or
  baking paper) so the reflection on the bottle is a soft wash, not a streak. Do not point a strip at the camera lens.
- **Black matte on everything the camera can see**: the wall behind the bottle for each camera (that is the other camera's wall, so
  both walls), the floor under the belt, and a small black hood/shroud around each lens.
- **Flicker:** use a constant-current DC supply, not a PWM dimmer. PWM strips make banding on rolling-shutter webcams and change
  brightness between frames, which the classifier reads as a defect.
- Keep the light level fixed (no daylight leak into the enclosure) -- the ROI and thresholds were learned on one lighting condition.

## 5. Camera settings: stop the autofocus / exposure "hunting"

Autofocus or auto-exposure drifting between frames is a classification-accuracy risk. The software already has the hooks:

- `settings.json` `camera_controls` = `{"<index>": {"focus": .., "exposure": .., "wb_temperature": .., "rotate": 90}}`.
  `infer.open_capture` switches auto modes off first and records what the driver read back in `infer.applied_controls`
  (shown in the Camera tab). SOFTWARE; fake-tested, **never tried on the EMEETs**.
- Procedure once mounted: (1) set manual focus on a bottle standing at the inspection point; (2) set manual exposure as short as the
  light allows -- motion blur is `speed x exposure` (100 mm/s x 1/250 s = 0.4 mm; 1/500 s = 0.2 mm); (3) lock white balance at 6500 K;
  (4) lock gain; (5) confirm in the Camera tab that the read-back values equal what you set (some UVC drivers ignore a control).
- After any change of focus/exposure/position: **re-run ROI calibration** (`calibrate.py`) and check Live; the ROI is measured per
  camera geometry. A camera moved after calibration crops the wrong region (historical bug #4 in CLAUDE.md).
- The EMEET Nova 4K is a consumer webcam with ISP processing (sharpening, HDR). Turn off any "AI framing / HDR / face tracking" in its
  driver if exposed; they change frame appearance from bottle to bottle.

## 6. Trigger, timing and the bottle queue

### Flow (SOFTWARE, as built)

```
photo-eye X0 -> ladder SET M2 -> PLCService trigger event
  -> Inspector: frames captured strictly AFTER the trigger, every line camera
  -> classifier/YOLO -> decision.decide() -> one PASS/REJECT/FAULT per bottle (inspection_id)
  -> FIFO with per-bottle deadline:  PASS -> M0 now;  REJECT -> M1 at (trigger + travel - T0)
  -> PLC times Y0 (reject cylinder) itself;  production CSV row
```

### Where the photo-eye goes (COMPUTED)

Let `D` = distance from the photo-eye to the camera optical axis, `v` = belt speed, `W` = `inspect_window_s`.
The bottle is centred in view at `t = D / v` after the trigger. Set

```
capture_delay_s  =  D / v  -  W / 2          (settings.json; frames older than trigger + this are ignored)
```

Example: D = 250 mm, v = 100 mm/s -> D/v = 2.5 s; with W = 0.6 s => `capture_delay_s` = 2.2 s. The photo-eye only needs to be far enough
upstream that `D/v >= W/2 + a margin`, i.e. > ~100 mm at this speed; you have far more room than that.

**Gap in the current software:** `capture_delay_s` is one value for *all* cameras. With the 120 mm stagger, camera B sees the bottle
`120 / 100 = 1.2 s` after (or before) camera A. Needed before staggered mounting works: **a per-camera delay**
(`settings.json` e.g. `camera_delay_s: {"2": 2.2, "3": 3.4}`, applied in `Inspector.collect`, `machine_cycle.py`). Small change, fake-testable;
not done yet. Alternative without code: mount the cameras at the same belt position and mask the opposing camera with a baffle.

### Reject timing

`travel_s = inspection_to_reject_mm / conveyor_mm_s` (else the PLC's T0). Both settings are **0 = not measured** today. Measure the
distance from the photo-eye (the trigger point) to the Festo cylinder push line, and `conveyor_mm_s`. `timing_problem()` blocks
Start line if T0 >= travel (every REJECT would be late). Example: 600 mm / 100 mm/s = 6 s travel.

### The queue (your "second bottle arrives" question)

- **Software side: handled.** Every trigger creates a bottle with its own `inspection_id`, its own deadline and its own result. A
  second bottle never overwrites the first bottle's decision; the FIFO pops by `inspection_id`. A REJECT that would miss its deadline is
  **not fired late**: the bottle is answered with M0 and recorded as FAULT, with an alarm to remove it by hand.
- **Inspection is serial** (one thread, ~0.6 s window + inference). Bottles must therefore be spaced by more than that: at 100 mm/s,
  **keep >= ~100 mm clear gap (~1 s)** between bottles, plus margin. Closer than that and the second bottle's frames overlap the first.
- **PLC side is the real limit** (the ladder is the owner's; `docs/roadmap/PLC_COMMUNICATION.md` section 0): it is a **one-bottle
  M0/M1/M2 handshake**, and triggers are **masked while M1 is held** (net 5 resets M2 every scan while M1 is ON). A bottle arriving
  at X0 during a reject cycle produces no M2; the software counts it as `BOTTLE_UNTRIGGERED` and makes it a FAULT "NOT INSPECTED"
  bottle (rejected by default) rather than losing it silently. The saved 09:06 ladder still has **T0 = K150 (15 s) / T1 = K50 (5 s)**
  against the stated K15 / K5 contract (1.5 s / 0.5 s). With 15 s, one REJECT blocks the line's triggers for ~15 s -- at ~1 bottle
  per 1.5 s that discards ~10 bottles. **Fix K150 -> K15 (and verify T1) in the ladder before running real bottles.** That is the
  owner's file; this repo never edits it.
- True multi-bottle-in-flight without a masked window needs a PLC-side FIFO (or a shift register of results). Deferred; see
  `docs/roadmap/FUTURE_ENHANCEMENTS.md`.

### Emergency stop / fault handling

- A **hardwired E-stop must cut actuator/conveyor power through the safety chain, not through the PC or the PLC program.** Software
  must never be the thing that makes the machine safe. Wire the E-stop contact *also* to a PLC input so the program (and the HMI)
  can display "E-STOP" and clear M0/M1/M2, but the power cut itself is in hardware.
- Software FAULT behaviour today (SOFTWARE): any camera/model/PLC/timing failure => FAULT, physically rejected by default
  (`fault_action: REJECT`). The PLC service never writes Y outputs; only M0 (PASS) and M1 (REJECT).
- **Not built:** an E-stop input (needs an X address assigned in the ladder), a PC<->PLC heartbeat/watchdog (so the PLC can stop the
  line if the PC dies), and fault latching. These are in the integration checklist below.

## 7. Commissioning order (what to do tomorrow, in this order)

1. Plug each camera into its own USB port (different root hub). Measure real FOV with a ruler at the inspection point.
2. Measure: belt speed, photo-eye -> camera axis (`D`), photo-eye -> reject cylinder (`inspection_to_reject_mm`), bottle size.
3. Fix the ladder presets (K150 -> K15 / K50 -> K5) and export the I/O list; confirm real Y0/Y1/X0 addresses against
   `plc/address_map.py` using `python -m plc.commissioning` (never trust a simulator address on the machine).
4. Mount camera A only at 1080p portrait; lock focus/exposure/WB; check in Production live view that cap, label and meniscus are all visible.
5. Run `calibrate.py`; capture 100+ bottles with the real lighting. **The current model was trained on other images; expect a
   domain shift** -- retrain/fine-tune on frames from this enclosure before trusting PASS/REJECT.
6. Add camera B, set per-camera delays, test with known good + deliberately defective bottles, and failure injection (unplug a
   camera, kill the PLC link, block the sensor).
7. Dry-run the reject actuator with the line empty, then with sacrificial bottles. Record all of it in
   `FINAL_YEAR_BLACKBOOK/17_Hardware_Commissioning/README.md`.
