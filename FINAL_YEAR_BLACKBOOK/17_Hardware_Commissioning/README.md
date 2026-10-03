# Hardware Commissioning: results log

Every physical test goes here, one file per test, named `YYYY-MM-DD_<phase>_<what>.md`, with its raw
outputs (survey.json, result.json, production CSV, photos) in a folder of the same name. Then add a
row to `../RESULTS_INDEX.md`. Only results recorded here may be called **PHYSICAL** in the black book.

Status on 2026-10-03: **no physical test has been run.**

## Test sheets

### Phase 1: camera rig
| # | Check | Command / action | Pass when | Result |
|---|---|---|---|---|
| 1.1 | Cameras on separate USB ports | `python bench.py --survey 2 3` | both 1920x1080 MJPG ≥ 25 fps together | |
| 1.2 | Camera locked | Camera tab → Lock camera → Save + read back | no control "IGNORED by the driver" | |
| 1.3 | Bottle fills the tall frame | rotate 90 in Camera tab, Live tab view | cap and base inside the frame | |
| 1.4 | ROI calibrated | `python calibrate.py` | ROI box tight on the bottle | |
| 1.5 | No reflections / LEDs in view | Live tab, 3 bottles | label readable, no white stripe | |

### Phase 2: rig dataset
| Class | Physical bottles | Positions each | Frames | Done |
|---|---|---|---|---|
| GOOD | ≥ 5 | ≥ 4 rotations | | |
| NO CAP (bare neck) | ≥ 5 | ≥ 4 | ≥ 50 | |
| NO LABEL | ≥ 5 | ≥ 4 | | |
| TILTED CAP / SKEWED LABEL / DAMAGED LABEL | ≥ 3 each | ≥ 4 | | |

Capture with Live tab → tick the defects → Capture (goes straight into `labels.csv`).

### Phase 4: PLC
| # | Check | Pass when | Result |
|---|---|---|---|
| 4.1 | Serial link (Machine tab, Real PLC) | status CONNECTED, X/Y/M read | |
| 4.2 | Ladder T0/T1 set (ISPSoft) | T0 < travel time, T1 ≈ 0.5 s | |
| 4.3 | Hand on X0 | M2 goes 1 | |
| 4.4 | Machine tab test PASS | M0 → 0, M2 → 0, C0 +1, Y0 stays 0 | |
| 4.5 | Machine tab test REJECT | Y0 pulses ≈ T1 after T0, C1 +1 | |

### Phase 5–6: bottles through the machine
| Run | Bottles (GOOD/DEFECT) | Correct PASS | Correct REJECT | FAULT | Not inspected | Missed reject | CSV |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
