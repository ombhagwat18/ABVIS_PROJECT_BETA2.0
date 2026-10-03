# System Audit — Bottle Defect Inspection (2026-10-03)

Scope: the whole repository at commit `f09fdba` (`main`), including the vision code, models, data, PLC layer, GUI,
docs and repo hygiene. Method: every self-check run; safety-critical code read line by line (`infer.py`,
`plc/service.py`, `plc/client.py`, `MachineTab` and `TrainTab` in `gui.py`, `dataset.py` config/split code); every
checkpoint's `metrics.json`, both project configs and `labels.csv` inspected; roadmap claims checked against the code.
Nothing was run on hardware; no PLC or camera was driven.

---

## 1. Verdict

**The software foundation is unusually strong for a project at this stage. The machine is not ready, and four issues
block a real inspection cycle.**

* **Strong:** fail-safe PASS/REJECT/FAULT logic, monotonic-clock freshness checks, at-most-once PLC commands,
  one train/serve crop path, scene-based splits, model provenance with sha256 checksums, and honest documentation
  that does not overclaim. All 18 software self-checks pass.
* **Blocking:** (1) the saved ladder turns the reject solenoid on for only about one PLC scan; (2) the ladder lets a
  bottle pass if Python never answers; (3) the camera verdict is not connected to the PLC; (4) neither model has been
  validated on images from the production camera.
* **Model quality is weaker than the headline numbers.** Every Stage 1 checkpoint reports validation macro-F1 = 1.0, but
  on the held-out test set the active model falsely rejects **10 of 19 good bottles** (GOOD recall 0.474).

| Area | Rating | One-line reason |
|---|---|---|
| Inference safety logic (`infer.py`) | Strong | Every unknown state ends in FAULT; well tested |
| PLC communication (`plc/`) | Strong (software) | At-most-once, guarded writes, acknowledgement; simulator only |
| PLC ladder (`final_year.isp`) | **Blocking** | Reject pulse is about one scan long; no fail-safe default |
| Vision → PLC integration | **Not built** | The Machine tab sends manual test commands only |
| Stage 1 classifier | Weak | Saturated validation, possible leak, poor GOOD recall on test |
| Stage 2 detector | Fair | Good test mAP50 (0.968) on a small, different-domain set |
| Data | Weak | 72 good images, 0 `missing_cap`, 2 `missing_label` scenes |
| GUI | Fair | Works; 2 real bugs; 2,964 lines in one file |
| Docs | Good, drifting | Honest, but 8+ statements are now out of date |
| Repo hygiene | Fair | Whitelist `.gitignore` works; dead code and ~3 GB of stray local data |

---

## 2. Self-check results (run 2026-10-03, Python 3.8.0, RTX 3050)

| Check | Result |
|---|---|
| `dataset.py`, `train.py --demo`, `infer.py`, `inspection_trace.py`, `detect.py`, `calibrate.py --demo`, `charts.py`, `bench.py`, `migrate.py --demo`, `vision_data.py` | all pass |
| `gui.py --selftest` | pass, **but it ran against `bottle_detection`: "9 images, 0 defects"** (see H4) |
| `python -m plc.test_simulation`, `-m plc.test_service`, `-m plc.commissioning --selftest`, `-m plc.handshake_test --fake` | all pass |
| `stage2_dataset/review_app.py --selftest` | not run; known to fail by design (CLAUDE.md) |
| `plc.handshake_test --real`, `plc.sim_comm_test` | not run (needs the ISPSoft simulator) |

The PLC tests fail with `ImportError` if run as `python plc/test_service.py`. They must be run with `python -m`.
CLAUDE.md's test list doesn't include them at all.

All of these are software checks with fake captures, fake models, or a fake PLC. None is evidence the hardware works.

---

## 3. Findings

Severity: **Critical** = blocks a correct machine cycle or makes it unsafe · **High** = wrong results or misleading
quality claims · **Medium** = real bug or risk with limited blast radius · **Low** = hygiene.

### Critical

**C1. The reject solenoid fires for about one PLC scan.**
In the ladder saved at 01:36, net 6 does `OUT Y0` and `RST M1` on the same rung. M1 enables T0, so on the next
scan T0 drops, Y0 turns off, and T1 can never time out. On the simulator, Y0 was ON in 0 of 187 samples on one
run and 1 of 126 on another (`PLC_COMMUNICATION.md` §0, verified from the decoded file). A real solenoid will not
actuate on a one-scan pulse. The earlier backups (`final_year_2026-10-2-*.~bak`) have a correct `T0 → T1 → Y0`
pulse design. → The ladder owner must restore or fix net 5/6. This is the first thing to fix.

**C2. The ladder is not fail-safe: no answer means the bottle passes.**
X0 sets M2 and then the ladder waits for M0 or M1 indefinitely. Nothing rejects a bottle when:
* Python crashes or hangs, or the link faults;
* the vision result is FAULT (there is no FAULT bit, so FAULT cannot be sent);
* a second bottle arrives within 15 s of a REJECT. M1 is held for the whole T0 (K150 = 15 s), so
  `PLCService._submit` correctly refuses with "a command bit is already ON". The second bottle gets no answer
  and passes.

A QC line must reject by default. → Add a ladder-side inspection timeout that rejects when no answer arrives (or
gives a fault output). Map vision FAULT to REJECT (or a dedicated bit). Shorten T0 to the real travel time, or track
several bottles in flight (a shift register or FIFO keyed on X0).

**C3. Nothing connects the camera verdict to the PLC.**
`MachineTab` shows "camera not linked yet: AI result: none" and only sends operator-armed test commands
(`gui.py:869`). There is also no per-bottle decision. At about 15 Hz, one bottle produces many independent frame
verdicts, and nothing maps an M2 trigger to "the frames of this bottle". `Camera.frames_since(ts)` and
`RECENT_FRAMES` are groundwork for this, but no consumer exists. → Build the trigger → inspection window → single
decision → `send_pass`/`send_reject` loop (Phase 2 of `IMPLEMENTATION_PLAN.md`).

**C4. Neither model has been validated on images from the production camera.**
The classifier was trained on 2537×1927 black-backdrop JPEGs. The detector was trained on 1780×1000 crops of
Iriun-viewer *screenshots* on a light backdrop. The two datasets share no images (`VISION_DATASET.md` §4), and the
dev cameras default to 640×480 (4:3). The only live detector test had no bottle in view, and it drew false `label`
and `bottle` boxes on a TV and a door at conf 0.25. → Capture a labelled set of real bottles on the final
camera/lighting/backdrop before tuning anything else.

### High

**H1. Stage 1 quality claims don't hold up on the held-out test set.**
All 10 checkpoints report validation macro-F1 = 1.0 (or close). The held-out test for the active model
`20260919-164511` (238 images, 23 scenes, `HELD_OUT_TEST_RESULTS.md`) shows:

| | Result |
|---|---|
| GOOD (all-clear) recall | **0.474**: 10 of 19 good bottles falsely rejected |
| `skewed_bottle` precision | 0.721 (12 FP) |
| `tilt_cap` precision | 0.688 (10 FP) |
| Exact-match accuracy | 89.1% |

On a line, that means about 1 in 2 good bottles rejected. Contributing factors:
* **Possible validation leak.** `dataset.scene_map` groups frames per folder only. 403 cross-folder image pairs are the
  same frame-run by its own rule (e.g. `+ve` ↔ `Skewed Label`, `Tilt Cap` ↔ `Water Level`). Every existing checkpoint
  was trained on that split.
* **Hand-tuned live thresholds as low as 0.05** (`damaged_bottle`, `skewed_bottle`) and 0.1 for three others. These
  push toward false rejects and haven't been validated on test or live frames.
* Macro-F1 excludes unscored classes, so "1.0" sits next to `missing_cap` and `missing_label` both at 0.0.

→ Retrain on the leak-free split from `vision_data.py`, choose thresholds on validation only, report test results,
and stop quoting validation F1.

**H2. The data is too thin for the classes the system claims to detect.**
1,143 images ≈ 112 bottles. Good: 72 images in 8 scenes. `missing_cap`: **0** (disabled). `missing_label`: 30 images
in **2** scenes, so it can't appear in all three splits. 0 multi-label rows: the multi-label design is untested on real
data. Viewed samples of `skewed_bottle` and `missing_label` look alike (`VISION_DATASET.md` §2). → Collect far more
good bottles (the class that decides false rejects), real `missing_cap`/`missing_label` scenes, and genuine
multi-defect bottles.

**H3. Training while cameras run applies the wrong thresholds (bug).**
`TrainTab.start` (`gui.py:1623`) never stops or reloads the cameras. At the end of `train.run()` it writes the
*new* checkpoint's `thresholds` and `active_model` to `config.json` (`train.py:435-437`). The camera thread re-reads
config on every scored frame (`infer.py:566`), but it still holds the *old* `Model`. From then on, live verdicts mix
old-model scores with new-model thresholds, and nothing warns about it. (`AnalysisTab.activate` handles this correctly
by calling `cams.load_model`.) Related: every training run promotes itself to the active model even when it scores worse.
→ Bind thresholds to the loaded model (keep them on `Model`, or reload the model when `active_model` changes). Make
promotion an explicit step.

**H4. The app opens on an empty project, and the GUI integration test runs against it.**
`projects/active.txt` = `bottle_detection` (9 inbox images, no defect columns, no thresholds, no model). Launching
`run.bat` lands on it, and `gui.py --selftest` reports "9 images, 0 defects". So the "closest thing to an
integration test" no longer exercises labelling, defect columns, or analysis against real data. → Switch back to
`om_bottle`, or make the self-test pin its project.

### Medium

**M1. The Machine tab event log freezes after 5,000 events (bug).** `PLCService._events` is a
`deque(maxlen=5000)` (`plc/service.py:125`), and the tab redraws only when `len(events)` changes (`gui.py:886`).
Once the deque is full its length stays 5000, so the log stops updating while the PLC keeps running. Each bottle
produces roughly 10-15 events, so this happens after a few hundred bottles. → Compare the newest event's
timestamp, or keep a running counter.

**M2. `serial_ports()`, `SerialTransport` and `SERIAL_FORMATS` are defined twice in `plc/client.py`**
(lines 113-191 and 192-267, an exact copy-paste). The second copy silently wins. A fix applied to the first copy
would be dead code. → Delete one.

**M3. Auto-reconnect is on by default in the GUI** (`plc_auto_reconnect` defaults to `True`, `gui.py:553`), while the
service design says "No automatic reconnect" (`plc/service.py:19`). Commands are never replayed, and a re-seen
trigger is flagged `after_reconnect`, so this is safe as written. But the stated policy and the actual behaviour
differ. → Decide which one is intended and document it.

**M4. Capture and inference share a thread.** Inference time lowers the capture rate. `frame_ts` is when `read()`
returned, not when the image was exposed, and DirectShow buffering staleness isn't measured. With a 1.0 s freshness
window, a frame can be "fresh" but show a bottle that has already moved on. (Documented in `CURRENT_SYSTEM.md`.)
→ Measure glass-to-verdict latency against a real conveyor before setting any timing.

**M5. An unexplained hang:** a camera reopened after a two-camera session delivered 7 frames, stalled, and the
process hung at exit (`CURRENT_SYSTEM.md`). The FAULT logic reported it correctly, but the cause is unknown. A hang
at exit on a line PC means a manual restart.

**M6. `config.json` is written non-atomically** (`dataset.save_config` uses `write_text`) while the camera
thread reads it at ~15 Hz. A torn read raises → "inference failed" → FAULT for one frame. It fails safe, but it can
produce spurious FAULTs. (`save_labels` already uses a temp file and rename; apply the same to config.)

**M7. `requirements.txt` omits `pyserial`** (needed for the "Real PLC (serial)" mode), and `ultralytics` is
commented out (needed for every YOLO mode). Both are imported lazily, so the app starts without them, and those
modes then fail with a clear error. → List them, or add a `requirements-machine.txt`.

### Low

* **Dead web app with unauthenticated write endpoints.** `app.py` exposes `/api/train`, `/api/upload/delete`,
  `/api/model/activate` and others with no auth. Nothing launches it. → Delete `app.py` + `index.html` (the
  web-vs-desktop decision is resolved).
* `torch.load(..., weights_only=False)` in `infer.py:168` runs pickled code from a checkpoint. Acceptable for
  local files you created; never load a checkpoint from elsewhere.
* `gc.disable()` app-wide (`gui.py:123`), with collection on the Tk thread only. It fixes a real freeze, but it's
  fragile: any new code that blocks the Tk loop also stops GC.
* About 3 GB of local, git-ignored duplicates: `All Datasets/` (990 MB, the pre-migration copy),
  `yolo_detection_dataset/` (487 MB), root `yolov8n.pt` / `yolov8s.pt`, three stray `.log` files, and two
  `.git/objects/*/tmp_obj_*` garbage files.
* `gui.py` is 2,964 lines (10 tab classes plus a HMI). `MachineTab` (about 420 lines) would split out cleanly.
* `FINAL_YEAR_BLACKBOOK/06_New_Dataset_Audit/audit_script.py` hard-codes `E:\MACHINE LEARNING PROJECT\...`.
* **From my push earlier today:** I tracked `vision_dataset/manifests/*` and `export_info.json` (the images and
  `hash_cache.json` stay ignored), but `VISION_DATASET.md` says `vision_dataset/` is "not in git". Either untrack
  them or update that line.

---

## 4. Documentation drift

The docs are honest about limits, but these statements are now false:

| Doc | Says | Reality |
|---|---|---|
| `CURRENT_SYSTEM.md`, `FEATURE_STATUS.md` | "No held-out test split" | Since the 2026-09-19 checkpoints there is a 238-image `test_paths` and `train.evaluate_test()` |
| `CURRENT_SYSTEM.md` | 9 checkpoints | 10 (`20261003-101425`, convnext_tiny, trained today) |
| `CURRENT_SYSTEM.md` | GUI has 9 tabs; PLC layer "not integrated with the … GUI" | 10 tabs, including **Machine** (PLC) and **Annotate** |
| `FEATURE_STATUS.md` | Mock PLC "Not started" | `FakePLC` / `FakeLadder` exist in `plc/test_simulation.py` |
| `PLC_COMMUNICATION.md` §A, `plc/client.py:7` | "SerialTransport: NOT written" | It exists (and twice, see M2); §later in the same doc says so |
| `CLAUDE.md` | 8 tabs; `App` at `gui.py:53`, `TABS` at `:101`, self-test at `:1836`/`:1913`; "~2.2k lines" | `App` at `:107`, `TABS` at `:168`, `selftest()` at `:2777`, entry at `:2961`; 2,964 lines; 10 tabs |
| `CLAUDE.md` | Test list | Missing the four `python -m plc...` checks |
| `VISION_DATASET.md` | `vision_dataset/` not in git | Manifests now tracked (see Low) |

---

## 5. What is solid (keep doing this)

* **Fail-safe inference.** `infer.decide` turns empty or NaN scores into FAULT. `Camera.inspection` gives PASS/REJECT
  only from a fresh score, and a superseded thread can't write into a new session. The self-tests cover every
  FAULT branch with a real thread and fake captures.
* **PLC command discipline.** One worker thread owns the link. The "answered" flag is set before the write, nothing
  is retried, preconditions are re-read immediately before the write, success requires the PLC's acknowledgement,
  and only M0/M1 are writable (`simulator_test_write` refuses non-loopback links).
* **One crop path for training and inference** (`dataset.model_input`), with lossless PNG cache, ROI scaling via
  `roi_frame`, and checkpoint-baked crop parameters for safe rollback.
* **Evaluation honesty.** Scene splits, disabled zero-positive classes, "unscored" instead of fake 0/1 scores,
  `val_paths`/`test_paths` recorded per checkpoint, and sha256 provenance for the detector.
* **Documentation that labels evidence** (VERIFIED / USER-STATED / INFERRED / FAKE / SIMULATOR) and never claims
  hardware success.

---

## 6. Recommended order of work

1. **Fix the ladder (C1, C2):** restore a real reject pulse; add reject-on-timeout; decide how FAULT maps to a PLC
   action; size T0 to the measured travel time, or handle several bottles in flight.
2. **Quick software fixes (H3, H4, M1, M2, M6):** each is small and contained.
3. **Capture production-domain data (C4, H2):** the final camera, lighting and backdrop; many good bottles; real
   `missing_cap`/`missing_label`; label it.
4. **Retrain Stage 1 on the leak-free split (H1)** and judge it on test GOOD recall and per-class precision, not
   validation F1. Re-check the detector on the same images.
5. **Build the machine loop (C3):** M2 trigger → frames after the trigger → one decision per bottle → `send_pass` /
   `send_reject` → record the result in `TraceStore`.
6. **Run the simulator handshake** (`python -m plc.handshake_test --real`), then the physical PLC over serial, then the
   first real cycle with known good and defective bottles, with timing measured.
7. **Refresh the docs (section 4)** and delete `app.py`/`index.html`.
