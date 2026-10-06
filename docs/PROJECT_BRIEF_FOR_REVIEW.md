# Project brief for review and enhancement

**Purpose of this file:** one self-contained document that explains the whole project, so a reviewer (a person or an
AI assistant such as ChatGPT) can understand it without reading the code, then suggest improvements. Status date:
**2026-10-06**. Everything marked *software-tested* was tested with a fake PLC, fake cameras and fake models. **Nothing
has been validated on the physical machine** except a read-only link to the PLC.

For the full system reference (every setting, page, command) see `docs/TEAM_HANDBOOK.md`.

If you are an AI assistant asked to review this: read sections 1-9, then answer the questions in section 10. Do not
assume anything not written here; where the brief says UNKNOWN or NOT MEASURED, say what you would measure.

---

## 1. What the system is

A bottle-inspection machine for a conveyor line, first for **250 ml Bisleri water bottles**. A camera inspects each
bottle; bad bottles (and bottles the system cannot judge) are pushed off the belt by a pneumatic cylinder. It is built
as **one Windows desktop application** (Python 3.8, CustomTkinter, OpenCV, PyTorch, Ultralytics) that also talks to a
**Delta DVP-SS2 PLC**.

Defects to catch: missing cap, tilted cap, damaged bottle, damaged label, missing label, skewed bottle, skewed
label, wrong water level.

## 2. The physical machine (as stated by the owner; mostly unmeasured)

| Item | Detail |
|---|---|
| Conveyor | about 5 ft (1524 mm), bottle transit 14-16 s (roughly 95-110 mm/s). **No encoder.** |
| Sensor | photoelectric bottle sensor (X0) |
| PLC | Delta DVP-SS2 (ISPSoft ladder, RS-232 ASCII 9600 7E1, COM5, station 1) |
| Actuator | Festo DSNU cylinder with a 5/2 solenoid valve (Y0), reject station downstream |
| Cameras | 2 x EMEET Nova 4K (USB). Both on one USB 2.0 hub: only one 1080p stream works at a time |
| Lighting / enclosure | 2 LED strips (6500 K), black matte enclosure 450 x 320 x 400 mm |
| Computer | Windows 11, Python 3.8, RTX 3050 Laptop GPU (4 GB), 16 GB RAM |

**Planned camera layout:** Camera 1 on one wall at the photo-eye; Camera 2 on the *opposite* wall, shifted toward the
exit, so the same bottle is seen from the other side a moment later.

## 3. How one bottle is handled

```
X0 (photo-eye) -> PLC sets M2 ("inspect now")
  -> computer takes frames AFTER the trigger from each camera (camera 2 at trigger + offset / belt speed)
  -> AI: classifier (defect scores) + YOLO detector (bottle / cap / label boxes)
  -> decision engine: votes over frames, fuses cameras, applies the "recipe"  ->  PASS / REJECT / FAULT
  -> PASS : write M0 at once
     REJECT / FAULT : write M1 at (trigger + travel time - T0)   [FAULT is rejected by default]
  -> the PLC runs T0, then drives Y0 (the cylinder) for T1 by itself
  -> one row per bottle in a SQLite database, plus an evidence picture
```

- `travel time = distance(camera -> cylinder) / belt speed`, both **measured** values (not yet measured).
- The computer **never writes the PLC outputs**; it writes only M0 (PASS) and M1 (REJECT). One answer per trigger, never
  retried, trigger marked answered before the write.
- A reject that would be late is **never fired**; the bottle is flagged FAULT for removal by hand.
- A bottle that reaches X0 while the PLC is busy gets no trigger; the system records it as **NOT INSPECTED**.
- Several cameras are fused FAULT > REJECT > PASS; images that cannot be matched to the right bottle by time give
  `CAMERA_ASSOCIATION_FAULT`.

## 4. The AI

| | Classifier | Detector |
|---|---|---|
| Model | EfficientNet-B0 (multi-label; 8 outputs, one per defect) | YOLOv8n (3 classes: bottle, cap, label) |
| Input | ROI crop of the bottle, 192 x 448 | whole frame |
| Active checkpoint | `20260919-164511` | `models/stage2_yolo/stage2_best.pt` (v1) |
| Missing cap | cannot detect it (0 usable training positives) | a bottle with **no cap box** = missing cap, via the recipe |

**Recipe:** a per-product JSON (`config.json` "inspection": anchor class + parts with `required`, `search`, `zone`) that
says what a complete product looks like. A new product needs data + a detector + a recipe, not new code.

**Operator view:** one steady word per bottle (GOOD / DEFECT: <name> / CHECKING / NO BOTTLE / FAULT), decided over
several frames and latched until the bottle leaves (`verdict.py`). Boxes and score bars are an engineer-only view.

**Improvement loop (built, not yet used on the machine):** Live *Auto-collect* saves one frame per decided bottle to
the review inbox with the model's prediction; corrections are stored as hard examples; training oversamples good
bottles (x5) and hard examples (x3). Unsure bottles (score just under the threshold) show **CHECK** instead of GOOD.

**Model lifecycle:** training never activates a model. CANDIDATE -> VALIDATED (needs a held-out test, real-camera numbers within limits: good called defective <= 2 %,
defective passed <= 1 %, >= 30 + 30 real bottles, and, for a classifier, a background-shortcut check) -> APPROVED ->
ACTIVE, with a deployment log and rollback (`model_registry.py`, `model_checks.py`).

## 5. Data

| Dataset | Content | Notes |
|---|---|---|
| Classifier project `om_bottle` | 1,167 images, 8 defect columns, multi-label, only **72 good** images | split by *scene* (near-duplicate video frames), never randomly; includes 22 white-background missing-cap images that teach a shortcut (see section 7) |
| Stage 2 detection set | 594 images, 39 scenes, bottle / cap / label boxes (1,994 boxes) | images are screenshots of the Iriun Webcam viewer, **not** direct camera frames; splits v1, v2 (tight cap boxes), v3 (missing-cap scene moved to train) |

## 6. Results so far (all offline, none on the real camera)

| Measure | Value |
|---|---|
| Classifier (active) held-out test, 238 images | macro-F1 0.935; with the old hand-tuned thresholds: 10 of 19 good bottles called defective, 0 defective passed |
| After calibrating thresholds from validation | 22 false alarms instead of 28, but 1 of 219 defective bottles passed; still 10 of 19 good flagged |
| Detector v1 test mAP50 / mAP50-95 | 0.968 / 0.660 (87 images, 7 scenes) |
| Detector v3 candidate test mAP50 / mAP50-95 | 0.971 / 0.803 (52 images; no missing-cap bottle left in its test set) |
| Missing cap, detector (validation, 6 bottles) | v1 0/6, v2 6/6, v3 5/6; on the 12 full-bottle bare necks v1 0/12 (v3 trained on them, so not evidence) |
| Self-tests | 25 pass (`python selfcheck.py --full`) |

## 7. Known problems (honest list)

1. **PLC timers are wrong for the software.** The ladder has T0 = K150 (15 s) and T1 = K50 (5 s); the software expects
   about 1.5 s and 0.5 s. A ladder simulation (`python -m plc.ladder_sim`) confirms Y0 fires 15 s after M1 and stays on 5 s.
2. **One-bottle PLC handshake.** M2 stays on until the PC answers, and M1 masks new triggers for T0 + T1, so bottles closer
   than that are NOT INSPECTED. With the current ladder, minimum spacing after a reject is 20 s.
3. **Missing PLC safety features:** no E-stop status input, no interlock between Y0 and the conveyor, no timeout if the
   PC stops answering, no watchdog. (The hardware E-stop must cut power by itself.)
4. **Missing cap:** every existing detector reads the green tamper ring of a bare neck as a cap (about 0.70
   confidence). The classifier trained on the white-background missing-cap images learned "white background = missing
   cap" (it flagged 25 of 25 *capped* white-background bottles).
5. **Good bottles called defective.** Only 72 good training images; thresholds alone cannot fix it.
6. **Domain shift:** all training images are from a different capture setup than the EMEET cameras in the enclosure.
7. **Two cameras on one USB 2.0 hub** give one high-resolution stream.
8. **Belt speed, distances, T0/T1 actually loaded in the PLC, exposure and lighting: NOT MEASURED.**
9. **Real latency (trigger -> decision -> command -> cylinder): NOT MEASURED.**

## 8. Decisions already made (do not re-litigate without a reason)

- Python never drives Y outputs; the ladder owns conveyor, delay and cylinder.
- Unknown is FAULT, never PASS; FAULT bottles are rejected.
- Time-based tracking now (no encoder), with an abstraction ready for an encoder later.
- Splits are by scene; the test set is only read, never used to choose thresholds or epochs.
- Proposals from a model are never training data until a person accepts them.
- SQLite (one file per project) for the production record.
- The user owns the ladder; this project reads it but never edits it.

## 9. Where things are

Entry points: `README.md` (overview), `docs/guides/PRODUCTION_HMI_AND_COMMISSIONING.md` (operate and commission),
`docs/hardware/PLC_LADDER_REQUIREMENTS.md` (what the PLC must do), `docs/roadmap/FEATURE_STATUS.md` (per-feature truth),
`CLAUDE.md` (every rule the code relies on).

| Topic | Files |
|---|---|
| Machine cycle and decision | `machine_cycle.py`, `decision.py`, `tracking.py`, `machine_state.py`, `verdict.py` |
| PLC | `plc/` (protocol, client, service, fake ladder, `ladder_check.py`, `ladder_sim.py`), `plc file/final_year/final_year.isp` |
| AI | `infer.py`, `detect.py`, `train.py`, `model_bench.py`, `calibrate_thresholds.py`, `model_registry.py` |
| Data | `dataset.py`, `annotate.py`, `autoannotate.py`, `stage2_dataset/`, `projects/om_bottle/` |
| Record and alarms | `production_store.py`, `production_export.py`, `alarms.py`, `applog.py` |
| Application | `gui.py`, `hmi.py`, `theme.py` |
| Tests | `selfcheck.py` (all self-tests), each module runs its own self-test |

## 10. Questions for the reviewer

Please answer with concrete, ordered recommendations and say how you would test each one.

1. **Safety and handshake.** Is the PC/PLC design sound (M0/M1/M2, one answer per trigger, no retries, late reject never
   fired)? What failure modes are missing? Propose the minimum ladder changes (answer timeout, watchdog, E-stop input,
   Y0 interlock) and the "protocol v2" that allows several bottles in flight (see PLC_LADDER_REQUIREMENTS.md section 3.4).
2. **Timing without an encoder.** Is time-based tracking at roughly 100 mm/s with a speed tolerance of 10 % reasonable?
   What belt-slip, jitter or spacing risks should be measured first? When would an encoder pay for itself?
3. **Missing cap.** Given the data problem in section 7, what is the best plan: capture protocol, number of bottles and
   poses, hard negatives (bare neck, tamper ring, partial cap), annotation, and which model (detector vs classifier vs
   segmentation vs anomaly detection)?
4. **Good bottles called defective.** How would you reduce false rejects without letting defective bottles through
   (more good data, class balance, calibration, ensembling, a "re-check" band)?
5. **Dataset and validation.** Is scene-based splitting enough? How should a held-out test set be built from the real
   EMEET frames? How many bottles are needed for a defensible accuracy claim?
6. **Two-camera design.** Is "camera 2 on the opposite wall, shifted toward the exit" a good choice? Lighting, glare,
   cross-illumination, and what each camera should be responsible for.
7. **Decision logic.** Is frame voting (60 % of at least 6 frames, then latch) and FAULT > REJECT > PASS sensible? What
   would you change for a line running faster than today?
8. **Architecture.** What would you simplify or split (the GUI is about 5,000 lines)? What technical debt is most
   dangerous? Is SQLite right for traceability, and what is missing (batch, operator, recipe version, audit trail)?
9. **Performance.** Detector about 30-40 ms, classifier about 15-35 ms on an RTX 3050; what would you measure
   end to end, and where is the likely bottleneck?
10. **Commissioning plan.** Review the 16-step physical commissioning order in
    `docs/guides/PRODUCTION_HMI_AND_COMMISSIONING.md`: what is missing or in the wrong order?
11. **Industrial readiness.** What is needed before this can be called production-grade (validation protocol, MSA/gauge
    study, change control, user roles, backup, uptime)?

## 11. What the reviewer can and cannot assume

- Can assume: the software behaves as described in sections 3-5 *in simulation* (24 self-tests).
- Cannot assume: any accuracy on real EMEET frames, any timing, the program actually loaded in the real PLC (the file in
  the repository was last saved 2026-10-04 and is read-only here), or that the conveyor/cylinder behave as stated.
