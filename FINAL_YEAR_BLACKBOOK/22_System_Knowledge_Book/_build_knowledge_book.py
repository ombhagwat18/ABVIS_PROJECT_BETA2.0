"""Builds the System Knowledge Book (.docx): everything about the system in one document, with pictures and process
flowcharts, for team-mates, senior reviewers and examiners, and as a source to copy into the black book.

    python FINAL_YEAR_BLACKBOOK/22_System_Knowledge_Book/_make_knowledge_figures.py      # figures first
    python FINAL_YEAR_BLACKBOOK/22_System_Knowledge_Book/_build_knowledge_book.py

Numbers come from the files (metrics.json, test_metrics.json, YOLO candidate files, labels.csv, production.db,
settings.json, the self-check record); nothing is typed by hand except dated facts quoted from the project docs.
Read-only for the project: it writes only into this folder.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
F20 = ROOT / "FINAL_YEAR_BLACKBOOK/20_Final_Report/figures"
F21 = ROOT / "FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/figures"

# data loaders, table helpers and the Doc class of the black-book builder (folder 20): one source for every number
_spec = importlib.util.spec_from_file_location("_bb", ROOT / "FINAL_YEAR_BLACKBOOK/20_Final_Report/_build_docs.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
# the active 20260919-164511 was copied back into projects/om_bottle/models on 2026-10-13; B.active already holds it,
# so the run list keeps only the 2026-10-08 retrain (as when the black book was built)
B.cls_runs[:] = [r for r in B.cls_runs if r["m"]["stamp"] != B.active["m"]["stamp"]]

SETTINGS = json.loads((ROOT / "settings.json").read_text())
CONFIG = json.loads((ROOT / "projects/om_bottle/config.json").read_text())
BRAND = json.loads((ROOT / "app_assets/brand/brand.json").read_text())
SELFCHECK_FILE = ROOT / "FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/data/selfcheck_2026-10-14.txt"
NAME = BRAND["name"]


class Book(B.Doc):
    def pic(self, path, caption, width=16.0):
        """Figure from any folder (the parent class only reads folder 20)."""
        path = Path(path)
        if not path.exists():
            self.p(f"[missing figure {path.name}]", italic=True)
            return
        self.fig_n += 1
        par = self.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(path), width=Cm(width))
        par.paragraph_format.keep_with_next = True
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.fig_n}: {caption}"); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = B.GREYC

    def save(self, name):
        self.d.save(HERE / name)            # the parent class saves into folder 20
        print("wrote", HERE / name)

    def qa(self, q, a):
        par = self.p(f"**Q: {q}**"); par.paragraph_format.space_after = Pt(1)
        self.p(f"A: {a}", align="j")


def real_frames_strip():
    """Three real overlays produced by the application's TEST INSPECTION, side by side."""
    ps = [("captures/GT_FULL_20261003-134634/img_002_f1_overlay.jpg"),
          ("captures/GT_NOCAP_20261003-134650/img_328_f1_overlay.jpg"),
          ("captures/CHECK_20261006-133038/cam2_f1_overlay.jpg")]
    ims = [cv2.resize(cv2.imread(str(ROOT / p)), (640, 480)) for p in ps]
    gap = np.full((480, 12, 3), 255, np.uint8)
    out = FIG / "k12_real_overlays.jpg"
    cv2.imwrite(str(out), np.hstack([ims[0], gap, ims[1], gap, ims[2]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def selfcheck_rows():
    rows = []
    for line in SELFCHECK_FILE.read_text().splitlines():
        if line.startswith("ok ") or line.startswith("FAIL"):
            parts = line.split()
            rows.append([" ".join(parts[1:-1]), "PASS" if parts[0] == "ok" else "FAIL", parts[-1]])
    return rows


def build():
    D = Book(f"{NAME} - System Knowledge Book")
    b0 = next(r for r in B.cls_runs if r["m"]["arch"] == "efficientnet_b0")
    b0lf = b0["t"]["test_leak_free"]
    act_lf = B.active["t"]["test_leak_free"]
    v8s = B.yolo["YOLOv8s v3"]["c"]
    v8n = B.yolo["YOLOv8n v3"]["c"]
    speed = float(SETTINGS.get("conveyor_mm_s") or 0)
    dist = float(SETTINGS.get("inspection_to_reject_mm") or 0)
    t0, t1 = float(SETTINGS["plc_t0_s"]), float(SETTINGS["plc_t1_s"])
    travel = dist / speed if speed else 0
    n_bottles = sum(B.finals.values())
    sc = selfcheck_rows()
    sc_pass = sum(1 for r in sc if r[1] == "PASS")

    # ================================================================ cover
    D.p()
    par = D.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(ROOT / "app_assets/brand/nexainspect_logo.png"), width=Cm(9))
    D.p()
    D.p("SYSTEM KNOWLEDGE BOOK", bold=True, size=24, align="c", color=B.NAVY)
    D.p(f"{NAME} ({BRAND['descriptor']}) - project ABVIS", size=13, align="c")
    D.p("AI bottle-defect inspection with PLC-controlled rejection on a conveyor", italic=True, size=12, align="c")
    D.p()
    D.p("Everything about the system in one document: what we are building and why, the machine, the software, the "
        "data, how the models were trained, the decision logic, accuracy and every test result, the work done so far "
        "with dates, what is still open, and answers to the questions examiners ask.", align="c")
    D.p()
    D.p(f"Software version {BRAND['version']}  |  status date {datetime.date.today():%d %B %Y}", bold=True, align="c")
    D.p("Final-year project, B.Tech Robotics & Automation Engineering", align="c")
    D.p()
    D.note("**Honesty rule used in this book.** Every result carries its evidence level: **SELF-TEST** (software with a "
           "fake PLC, fake cameras or fake models), **HELD-OUT TEST** (images never used for training or selection), "
           "**REAL CAMERA** (frames from the production cameras) or **REAL LINE** (the physical conveyor and PLC). "
           "A software self-test is never evidence that the machine works.")
    D.brk()

    # ================================================================ how to use
    D.h("How to use this book", 1)
    D.table(["Reader", "Read first", "Then"], [
        ["Examiner / external reviewer", "1 Summary, 9 Accuracy and results, 12 Status", "14 Questions and answers"],
        ["Senior engineer / guide", "1 Summary, 4 Architecture, 5 Process flow, 8 Decision logic", "12 Status, 13 Roadmap"],
        ["Team-mate writing the black book", "Appendix A (where each chapter goes)", "copy text, tables and figures"],
        ["New team member", "1, 3, 5, 10, 15 (how to run)", "Appendix B settings, Appendix C files"]],
        "Who should read what", [4.5, 6.5, 5])
    D.p("All figures are PNG / JPG files in FINAL_YEAR_BLACKBOOK/22_System_Knowledge_Book/figures (new diagrams), "
        "20_Final_Report/figures (results) and 21_Production_Software_Plan/figures (plan flowcharts), so they can be "
        "pasted into slides or the black book directly. Rebuild this document with the two scripts named in "
        "README.md; the numbers update themselves.", align="j")
    D.h("Table of Contents", 1)
    D.toc()
    D.p("(In Word: right-click the table above and choose Update Field to fill in page numbers.)", italic=True, size=9)
    D.brk()

    # ================================================================ 1 summary
    D.h("1. Executive summary", 1)
    D.p(f"**What it is.** {NAME} is a machine-vision inspection station for a bottle conveyor. The first product is "
        "the 250 ml Bisleri water bottle. A photo-eye tells a Delta PLC that a bottle has arrived. The PLC tells a "
        "Windows PC, and the PC looks at the bottle with one or two USB cameras and two AI models. It decides PASS, "
        "REJECT or FAULT and tells the PLC. The PLC fires a pneumatic cylinder that pushes rejected bottles off the "
        "belt at the right moment. Every bottle is recorded.", align="j")
    D.p("**Defects targeted:** missing cap, tilted cap, damaged bottle, damaged label, missing label, skewed bottle, "
        "skewed label, wrong water level.", align="j")
    D.p("**Design principle: unknown is never good.** A broken camera, a crashed model, a stale image or a bottle no "
        "camera saw becomes FAULT, and a FAULT bottle is rejected. The PC never switches a PLC output itself; it only "
        "writes a PASS bit (M0) or a REJECT bit (M1).", align="j")
    D.table(["Area", "Where it stands", "Evidence"], [
        ["Software (data tools, training, registry, decision engine, machine cycle, PLC link, HMI, database)",
         f"built; {sc_pass}/{len(sc)} automated self-tests pass", "SELF-TEST"],
        ["Best classifier: EfficientNet-B0 (8 defects, multi-label)",
         f"macro-F1 {b0lf['macro_f1']:.3f} on {b0lf['n_images']} leak-free test images; "
         f"{b0lf['false_pass']} defective bottles passed, {b0lf['false_reject']} good bottles rejected", "HELD-OUT TEST"],
        ["Best detector: YOLOv8s (bottle / cap / label)",
         f"mAP50 {v8s['test']['map50']:.3f}, mAP50-95 {v8s['test']['map50_95']:.3f}", "HELD-OUT TEST"],
        ["Models on the production camera", "classifiers call every real frame defective; detector finds bottles "
         "but caps are unreliable", "REAL CAMERA"],
        ["First real-line trial", f"{n_bottles} bottles: {B.finals.get('PASS', 0)} PASS, {B.finals.get('REJECT', 0)} "
         f"REJECT, {B.finals.get('FAULT', 0)} FAULT (most not inspected: one-bottle PLC handshake)", "REAL LINE"],
        ["Safety behaviour on the real line", "every uninspected bottle recorded as FAULT; halt latch worked",
         "REAL LINE"]], "The project in one table", [6, 7.2, 2.8], size=8.5)
    D.p("**The honest conclusion.** The software system is complete and behaves safely. The AI models score well on "
        "their own test images but do not yet transfer to the production camera (a different camera, background and "
        "framing). The PLC program handles one bottle at a time. The next stage is production-camera data, retraining, "
        "a multi-bottle PLC queue and physical commissioning. Chapters 12 and 13 list these items.", align="j")
    D.pic(FIG / "k02_project_lifecycle.png", "The whole project as one loop", 15.5)

    # ================================================================ 2 problem
    D.h("2. Problem, objectives and scope", 1)
    D.h("2.1 The problem", 2)
    D.p("Bottling lines must check every container for cap, label, body and fill-level defects. Human inspectors "
        "tire, disagree with each other and cannot keep pace with the line. A defective bottle that reaches a "
        "customer costs far more than one removed on the line. The task: inspect every bottle automatically, remove "
        "the bad ones at the right moment, never pass a bottle that could not be inspected, and keep a traceable "
        "record.", align="j")
    D.h("2.2 Objectives", 2)
    D.numbered(["Build a labelled multi-label defect dataset with a leak-free, scene-based split.",
                "Train and compare CNN classifiers; select on a held-out test set.",
                "Train a component detector (bottle, cap, label) for presence checks such as a missing cap.",
                "Build a decision engine that turns multi-frame, multi-camera evidence into PASS / REJECT / FAULT.",
                "Integrate the PLC: trigger, PASS / REJECT handshake, reject timing without an encoder.",
                "Provide an operator HMI, alarms, a model registry and a production database.",
                "Verify with automated self-tests, then trial on the real line and report the result honestly."])
    D.h("2.3 Scope", 2)
    D.p("In scope now: one product, up to two cameras, one reject station, time-based tracking. Deliberately "
        "deferred: encoder tracking, multi-product changeover, OCR / barcode, anomaly detection, user login, remote "
        "monitoring, certified safety functions (the hardware E-stop must cut power by itself).", align="j")

    # ================================================================ 3 machine
    D.h("3. The machine (hardware)", 1)
    D.pic(FIG / "k01_machine_layout.png", "Machine layout, top view")
    D.table(["Component", "Detail"], [
        ["Conveyor", f"about 1524 mm (5 ft); belt speed {speed:.1f} mm/s (saved value, no calibration record yet); "
                     "no encoder"],
        ["Bottle sensor", "photoelectric sensor on PLC input X0; start / stop buttons on X1 / X2"],
        ["PLC", "Delta DVP-SS2, ladder written in ISPSoft; RS-232 Modbus ASCII 9600 7E1, COM5, station 1"],
        ["Reject actuator", "Festo DSNU pneumatic cylinder with a 5/2 solenoid valve on output Y0; conveyor motor Y1"],
        ["Cameras", "2 x EMEET SmartCam Nova 4K (USB UVC), 1920 x 1080 MJPG. On one USB 2.0 hub only one 1080p "
                    "stream works (measured 2026-10-03): they need separate USB 3 ports"],
        ["Lighting / enclosure", "2 LED strips 6500 K, black matte enclosure 450 x 320 x 400 mm"],
        ["Computer", "Intel Core i7-12700H, 16 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11, Python 3.8"],
        ["Distances", f"inspection -> reject cylinder {dist:.0f} mm, so travel time = {dist:.0f} / {speed:.1f} = "
                      f"{travel:.1f} s"]], "Hardware", [3.5, 12.5], size=9)
    D.h("3.1 PLC device map", 2)
    D.table(["Device", "Meaning", "Who writes it"], [
        ["X0", "photo-eye: bottle at the inspection station", "field"],
        ["X1 / X2", "start / stop buttons (SET / RST conveyor Y1)", "field"],
        ["M2", "PLC -> PC: inspect now (rising edge of X0)", "PLC ladder"],
        ["M0", "PC -> PLC: PASS (PLC clears it and M2)", "PC"],
        ["M1", "PC -> PLC: REJECT (held through the reject cycle)", "PC"],
        ["T0 / T1", f"delay before firing / cylinder pulse (now {t0:.1f} s / {t1:.1f} s)", "PLC timers"],
        ["Y0 / Y1", "reject cylinder / conveyor motor", "PLC only - the PC never writes Y"],
        ["C0 / C1", "PASS / REJECT counters", "PLC"],
        ["M10 / M11", "commissioning start / stop pulses (off unless plc_operator_controls)", "PC, opt-in"]],
        "PLC device map (plc/address_map.py)", [2.2, 9.3, 4.5], size=9)
    D.h("3.2 The ladder program (7 networks)", 2)
    D.table(["Net", "Logic", "Meaning"], [
        ["1", "X1 -> SET Y1", "start button latches the conveyor on"],
        ["2", "X2 -> RST Y1", "stop button"],
        ["3", "X0 rising edge -> SET M2", "a bottle arrived: ask the PC to inspect"],
        ["4", "M0 -> RST M2, RST M0, CNT C0", "PASS answered: clear and count"],
        ["5", f"M1 -> TMR T0 K{int(t0 * 10)}, RST M2, CNT C1", "REJECT: start the delay, clear the trigger, count"],
        ["6", f"T0 -> TMR T1 K{int(t1 * 10)}, OUT Y0", "delay done: fire the cylinder for T1"],
        ["7", "T1 -> RST M1, RST M2", "reject cycle finished"]],
        "Ladder networks (plc file/final_year/final_year.isp, decoded by plc.ladder_check)", [1.2, 6.3, 8.5], size=9)
    D.p(f"The ladder belongs to the user and is never edited by the software. It is decoded and run in a scan "
        f"simulator (python -m plc.ladder_sim) against 14 scenarios. Its structure is correct and the timers now "
        f"match the settings (K{int(t0 * 10)} / K{int(t1 * 10)} at 100 ms = {t0:.1f} s / {t1:.1f} s). Still missing: "
        "an E-stop status input, a Y0 / conveyor interlock, a timeout if the PC stops answering, a heartbeat, and a "
        "multi-bottle queue. While M1 is held, network 5 keeps clearing M2, so a bottle arriving during a reject cycle "
        "gets no trigger.", align="j")
    D.pic(FIG / "k06_plc_timing.png", "PLC handshake timing for one PASS and one REJECT bottle")
    D.p(f"**Worked example.** Bottle B is triggered at t = 2.0 s. Travel is {travel:.1f} s, so it reaches the cylinder "
        f"at {2 + travel:.1f} s. The PC writes M1 at trigger + travel - T0 = 2.0 + {travel - t0:.1f} = "
        f"{2 + travel - t0:.1f} s. The PLC counts T0 ({t0:.1f} s) and fires Y0 at {2 + travel:.1f} s, exactly when "
        f"the bottle arrives, for T1 = {t1:.1f} s. Until {2 + travel + t1:.1f} s no new trigger is possible, so "
        "bottle C is recorded NOT INSPECTED. If the PC's decision were ready later than the M1 send time, the reject "
        "would be late, so it is never fired. The bottle becomes FAULT and an alarm tells the operator to remove it "
        "by hand.", align="j")

    # ================================================================ 4 architecture
    D.h("4. System architecture", 1)
    D.pic(F20 / "fig01_architecture.png", "System architecture")
    D.table(["Package", "Responsibility"], [
        ["vision/", "dataset and labels, training, inference (Camera, Model), YOLO detector, ROI calibration, "
                    "threshold calibration, stable verdict, annotation and auto-annotation"],
        ["line/", "decision engine, machine cycle (FIFO + deadlines), time-based tracking, machine state, alarms, "
                  "production database, export, event logs, the one clock"],
        ["registry/", "model registry (candidate > validated > approved > active), benchmarking, activation gates"],
        ["plc/", "Modbus ASCII protocol, address map and write policy, client, PLCService, ladder decoder and "
                 "simulator, fake PLC for tests"],
        ["ui/", "main window and 15 pages, HMI pages, theme, charts, Annotation Studio, brand"],
        ["tools/", "self-check runner, system report, model archive, brand assets, migration"]],
        "Code organisation (about 28,000 lines of Python)", [2.6, 13.4], size=9)
    D.h("4.1 Rules the design relies on", 2)
    D.bullets(["**One crop path.** dataset.model_input is used by both training and live inference (no train/serve skew).",
               "**One decision layer.** line/decision.py; AI models never command the PLC.",
               "**One owner of the PLC link.** PLCService does every transaction on one thread.",
               "**One machine state** (machine_state.py) and **one steady verdict per bottle** (verdict.py) on screens.",
               "**One clock.** line/clock.py = time.perf_counter (100 ns); time.monotonic ticks every 15.6 ms on this "
               "Windows / Python 3.8 and was replaced.",
               "**AI off the deadline thread.** Models run on a worker, so the machine cycle never waits for a model.",
               "**Configuration over code.** A new product needs data, a detector and a recipe (JSON), not code.",
               "**Tk is single-threaded.** Background work reports through App.post / App.run_bg."])
    D.pic(FIG / "k09_threads.png", "Threads of the running application", 15)

    # ================================================================ 5 process
    D.h("5. Process flow: one bottle from sensor to record", 1)
    D.pic(F21 / "fig_plan_02_bottle_cycle.png", "Per-bottle production flowchart with every fault branch", 14.5)
    D.numbered([
        "Operator presses START. The start checklist must be green (PLC in RUN, cameras, models, timing, FAULT -> "
        "REJECT, E-stop released, recipe valid). Models load, cameras open and must deliver fresh frames.",
        "Bottle breaks the photo-eye X0; the ladder sets M2. PLCService sees it (fast poll) and stamps the trigger "
        "with its time window (+/- uncertainty).",
        "The Inspector opens each camera's frame window (camera 2: trigger + offset / speed) and takes only frames "
        "stamped strictly after the trigger.",
        "The AI worker scores the frames: classifier (8 defect scores) and YOLO detector (bottle, cap, label boxes).",
        "decision.decide() applies the recipe and the three-zone rule, votes over frames and fuses cameras: PASS, "
        "REJECT or FAULT.",
        "PASS: M0 is written at once. REJECT / FAULT: M1 is scheduled at trigger + travel - T0. The trigger is marked "
        "answered before the write; nothing is retried.",
        "The PLC clears the bit (acknowledge), times T0 and fires Y0 for T1.",
        "One row in SQLite (decision, defects, scores, models, timings, PLC outcome, evidence image path), the daily "
        "CSV and the event logs; counters on the HMI update."])
    D.pic(FIG / "k07_machine_states.png", "Machine states", 15.5)

    # ================================================================ 6 data
    D.h("6. Data", 1)
    D.h("6.1 Classification dataset (project om_bottle)", 2)
    D.p(f"{len(B.lab) - 1} images, {len(B.lab_rev)} reviewed, about {B.n_scenes} independent scenes (physical bottle "
        "passes). labels.csv is the single source of truth: one row per image, one column per defect, and a reviewed "
        "flag. GOOD means every defect column is 0, so 'good' can never contradict a defect.", align="j")
    B.t_dataset(D, "Classification dataset per class")
    D.pic(F20 / "fig04_dataset_distribution.png", "Images and distinct scenes per class")
    D.pic(F20 / "fig05_dataset_samples.png", "One training image per class")
    D.pic(FIG / "k11_dataset_pipeline.png", "How folders, labels and splits fit together", 15.5)
    D.h("6.2 Why the split is by scene", 2)
    D.p("The images are consecutive video frames, so many are near-identical. A random split would put copies of the "
        "same bottle in training and test and report false accuracy. Frames are grouped into scenes and whole scenes "
        "go to train / validation / test (about 70 / 15 / 15). The test set is scored once, after every decision. "
        "A stricter 'leak-free' test subset also removes images that a cross-folder audit linked to training scenes.",
        align="j")
    D.h("6.3 Pre-processing", 2)
    D.pic(FIG / "k03_preprocessing.png", "A real training image through the pre-processing chain")
    D.p("The region of interest (ROI) is measured once from Sobel edge energy, because the clear bottle is nearly as "
        "dark as the backdrop. The input is tall (192 x 448), not square: letterboxing a tall bottle into a square "
        "would waste half the pixels on padding, exactly where the water level and label print are. The crop cache is "
        "lossless PNG (JPEG moved water_level scores by about 0.15). The ROI is stored inside each checkpoint, so "
        "rolling back to an old model always feeds it the crop it was trained on.", align="j")
    D.h("6.4 Detection dataset (Stage 2)", 2)
    D.p("594 reviewed images, 39 scenes, 1,994 boxes of three classes (bottle, cap, label). Split v3 = 453 train / "
        "89 validation / 52 test images. v3 moved the only full-bottle missing-cap scene into training, so the v3 "
        "test set has no missing-cap bottle and missing-cap detection must be validated on the real machine. The "
        "dataset's provenance (annotations, splits, hashes) is tracked; the model's MODEL_PROVENANCE.json pins the "
        "annotation file by sha256.", align="j")
    D.h("6.5 Data limitations found", 2)
    D.bullets(["Only 72 good images from 8 scenes; missing_label from 2 scenes; no image carries two defects.",
               "The 22 missing-cap pictures are one bottle on a white background. A classifier trained on them "
               "learned 'white background = missing cap' (25 / 25 capped white bottles flagged) and was rejected.",
               "All training images come from another camera and set-up than the EMEET production cameras."])
    D.pic(F20 / "fig06_domain_gap.png", "Domain gap: training images vs the production camera")

    # ================================================================ 7 training
    D.h("7. Model training", 1)
    D.h("7.1 Stage 1: multi-label defect classifier", 2)
    D.pic(FIG / "k04_training_pipeline.png", "How a classifier checkpoint is trained and chosen", 13.5)
    D.p("Multi-label means each defect has its own sigmoid output and its own threshold: a bottle can be skewed and "
        "have a damaged label at once. pos_weight balances rare defects. No horizontal flip is used, because it "
        "would mirror the label text and invert the skew direction.", align="j")
    D.note("The threshold rule (middle of the best-F1 band), temperature calibration, validation-loss tie-break and "
           "robustness augmentation in Figure 15 were added on 2026-10-13. The checkpoints of 2026-10-08 in the "
           "tables below were trained before that, with the older rule that picked the lowest tied threshold (often "
           "0.05; see Chapter 9).")
    B.t_epochs(D, "Training runs of the current models (2026-10-08 retrain)")
    D.pic(F20 / "fig07_cls_training_curves.png", "Classifier training curves")
    D.pic(F20 / "fig12_epochs.png", "Epochs run and epoch kept, every current model")
    D.p("Validation macro-F1 reaches 1.000 by about epoch 12 for every architecture. The validation set therefore "
        "cannot separate epochs or thresholds, which is why the held-out test set (Chapter 9) is the number to quote.",
        align="j")
    D.h("7.2 Stage 2: component detector (YOLOv8)", 2)
    D.p("YOLOv8n (3.0 M parameters) and YOLOv8s (11.1 M) were fine-tuned from COCO weights on split v3: 640 px, "
        "batch 8, at most 80 epochs, early stop after 20 epochs without improvement, workers = 0 (more workers "
        "exhausted the 16 GB machine). The best epoch is picked on validation; test is evaluated once. YOLOv8s was "
        "still improving at epoch 80.", align="j")
    D.pic(F20 / "fig14_yolo_curves.png", "Detector training curves")
    D.pic(F20 / "fig16b_yolo_results.png", "Ultralytics training summary, YOLOv8s v3", 15)
    D.h("7.3 Stage 3: segmentation", 2)
    D.p("A segmentation interface exists (vision/segment.py, label area and fill rules) and 546 label polygons are "
        "seeded, none reviewed. No segmentation model is trained, so any recipe that needs it returns FAULT.",
        align="j")
    D.h("7.4 Model lifecycle: nothing reaches production by itself", 2)
    D.pic(F21 / "fig_plan_03_model_lifecycle.png", "Data and model lifecycle", 13)
    D.table(["Gate (model_checks.py)", "Limit"], [
        ["Good bottles called defective (real camera)", "<= 2 %"], ["Defective bottles passed (real camera)", "<= 1 %"],
        ["Real-camera sample", ">= 30 good + 30 defective bottles"],
        ["Background-shortcut check (classifier)", "fires on <= 10 % of capped white-background bottles"],
        ["Held-out test result", "required before VALIDATED"]], "Activation gates", [9, 7], size=9)

    # ================================================================ 8 decision
    D.h("8. Decision logic", 1)
    D.pic(FIG / "k05_decision_logic.png", "From frames and cameras to one verdict", 14.5)
    D.pic(FIG / "k08_recipe_zones.png", "How the detection recipe turns boxes into findings", 15)
    rules = {**{k: v for k, v in (SETTINGS.get("decision_rules") or {}).items()}}
    D.table(["Rule", "Value used on the line", "Meaning"], [
        ["vote", "majority", "a defect must appear in most of the bottle's frames"],
        ["pass_max", "0.10", "a score must be at or below this to count as 'absent'; between it and the threshold = UNSURE"],
        ["min_bottle_frames_frac", "0.6", "the bottle must be seen in >= 60 % of a camera's frames, else FAULT"],
        ["det_min_conf_by_class", json.dumps(rules.get("det_min_conf_by_class", {})).replace('"', ""),
         "boxes below this confidence are ignored"],
        ["bottle_min_aspect / min_area", "1.3 / 2 %", "what counts as an upright bottle"],
        ["cap_top_fraction / above_margin", "0.5 / 0.25", "cap must sit in the top half of the bottle box (or just above)"],
        ["fault_action", str(SETTINGS.get("fault_action")), "what happens to a FAULT bottle (START refuses PASS)"],
        ["fault_latch_after", str(SETTINGS.get("fault_latch_after", 3)), "consecutive FAULT bottles that halt the line"],
        ["inspect_frames / window", f"{SETTINGS.get('inspect_frames')} frames / {SETTINGS.get('inspect_window_s')} s",
         "frames taken per camera per bottle"]],
        "Decision rules in force (decision.RULES + machine_cycle.PRODUCTION_RULES + settings.json)", [4, 3.4, 8.6],
        size=8.5)
    D.p("Classifier thresholds in the project config: " + ", ".join(f"{k} {v}" for k, v in CONFIG["thresholds"].items())
        + ". A threshold of 1.01 can never be reached (scores are 0-1), so that defect is switched off in the "
          "classifier. Missing cap is judged by the detector recipe instead.", align="j")
    D.h("8.1 What the operator sees", 2)
    D.p("Operators never see per-frame scores. verdict.py shows one steady word per bottle: GOOD, DEFECT: <name>, "
        "CHECK: <defect>? (unsure), NO BOTTLE or FAULT. It ignores the first frames, needs 6 valid frames, keeps a "
        "defect seen in >= 60 % of them, then latches the result until the bottle leaves. The PLC, database and code "
        "keep PASS / REJECT / FAULT.", align="j")
    D.h("8.2 Safety rules that are never weakened", 2)
    D.numbered(["Unknown is FAULT, never PASS; a FAULT bottle is rejected.",
                "The AI never commands the PLC; only the decision engine does, through one PLCService.",
                "The PC writes only M0 / M1 (plus opt-in commissioning bits), never X or Y devices.",
                "One answer per trigger; never retried; the trigger is marked answered before the write.",
                "A late reject is never fired; the bottle becomes FAULT with a remove-by-hand alarm.",
                "The real PLC never connects automatically; physical test commands need engineer + test mode + arm.",
                "The software HALT latch stops answering; the hardware E-stop must cut power by itself.",
                "A model reaches production only through the registry gates; training never activates a model.",
                "A model proposal is never a label until a person accepts it."])

    # ================================================================ 9 accuracy
    D.h("9. Accuracy and results", 1)
    D.h("9.1 How accuracy is measured", 2)
    D.table(["Measure", "Meaning"], [
        ["Precision", "of the bottles flagged with a defect, the share that really have it"],
        ["Recall", "of the bottles that have a defect, the share that were flagged"],
        ["F1", "harmonic mean of precision and recall (one number per defect)"],
        ["Macro-F1", "average F1 over all defects, each defect weighted equally (rare defects count as much as common)"],
        ["Exact match", "share of images where every one of the 8 defect flags is right"],
        ["PASS / REJECT accuracy", "share of bottles with the right accept / reject decision"],
        ["False PASS", "defective bottle accepted - the dangerous error"],
        ["False REJECT", "good bottle rejected - costs product, not safety"],
        ["mAP50", "detector: mean average precision with a box counted correct at >= 50 % overlap (IoU)"],
        ["mAP50-95", "the same averaged over overlaps 50-95 %: rewards tight boxes"]], "Metrics used", [3.8, 12.2], size=9)
    D.note("**HELD-OUT TEST** results below. The test set was never used to train, pick an epoch or set a threshold.")
    D.h("9.2 Classifier", 2)
    B.t_cls_test(D, "Classifier held-out test results (all checkpoints)")
    D.pic(F20 / "fig08_cls_test_f1.png", "Classifier test macro-F1")
    D.pic(F20 / "fig09_cls_perclass_heatmap.png", "Per-class F1 on the leak-free test set")
    B.t_perclass(D, "EfficientNet-B0 (20261008-081732) per defect, leak-free test set")
    D.pic(F20 / "fig10_cls_errors.png", "False PASS and false REJECT per model")
    D.pic(F20 / "fig11_cls_speed_accuracy.png", "Accuracy vs inference time and model size", 12)
    D.p(f"EfficientNet-B0 is the best architecture. The 2026-10-08 B0 reaches macro-F1 {b0lf['macro_f1']:.3f} "
        f"and the active checkpoint 20260919-164511 reaches {act_lf['macro_f1']:.3f}; both pass no defective bottle. "
        f"The weak point is good bottles: {act_lf['false_reject']} false REJECTs for the active model. The cause is "
        "the 72 good training images; better thresholds alone cannot fix that. tilt_cap is the weakest defect for "
        "every architecture, because a small angle is hard to read from a whole-image score.", align="j")
    D.pic(F20 / "fig13_thresholds.png", "Thresholds chosen by training vs the three-zone limits")
    D.h("9.3 Detector", 2)
    B.t_yolo(D, "Detector candidates, validation and test")
    B.t_yolo_perclass(D, "Detector per-class test results")
    D.pic(F20 / "fig16_yolo_pr_curve.png", "Precision-recall curve, YOLOv8s v3, test split", 12)
    D.pic(F20 / "fig17_yolo_confusion.png", "Normalised confusion matrix, YOLOv8s v3, test split", 12)
    D.h("9.4 Missing cap", 2)
    D.p("Detectors v1 and v2 boxed the green tamper ring of a bare neck as a cap (about 0.70 confidence) and missed "
        "all 12 full-bottle missing-cap images. v3 trained on that scene, so its score on it is not proof. The middle "
        "picture below shows the effect.", align="j")
    D.pic(real_frames_strip(), "Real overlays from the application. Left: complete bottle, PASS. Middle: bare neck "
          "and no label; the tamper ring is boxed as a cap, so only missing_label is reported. Right: production camera "
          "2 pointed at a cable, where the classifier invents defects.")
    D.h("9.5 Real production-camera frames (REAL CAMERA)", 2)
    D.p("Every model was run on 33 frames from the EMEET production cameras: bottles on the conveyor, the empty belt, "
        "an empty desk and a cable close-up. None shows a defective bottle.", align="j")
    B.t_real(D, "Classifiers on real production-camera frames")
    D.pic(F20 / "fig19_real_camera_classifiers.png", "Classifiers on real production-camera frames")
    D.pic(F20 / "fig20_real_camera_detectors.png", "Detectors on real production-camera frames")
    D.pic(F20 / "fig18_real_camera_detections.jpg", "YOLOv8s v3 detections on all 33 real frames")
    D.p("Every classifier calls every real frame defective, including empty scenes. YOLOv8s v3 behaves best: no "
        "boxes on empty scenes, bottles found at 0.82-0.89. But it misses caps that are cut off at the top of camera "
        "2's view, and it draws a 0.92 'cap' on a red cable. Lesson: test scores measure how well a model fits its "
        "own image distribution, not a new camera.", align="j")
    D.h("9.6 First real-line trial (REAL LINE)", 2)
    B.t_production(D, f"Real-line trial 2026-10-06/07 (production.db, {n_bottles} bottles)")
    D.pic(F20 / "fig21_production_trial.png", "Real-line trial outcomes")
    D.pic(F20 / "fig03_throughput_budget.png", "Why bottles went uninspected: one-bottle handshake time budget")
    D.p("Most bottles were never inspected: the ladder raises a new trigger only after the previous bottle is "
        "answered, and a REJECT masks triggers for T0 + T1. The safety design worked as intended: every uninspected "
        "bottle was recorded as FAULT, and the latch halted the line after three consecutive FAULTs.", align="j")
    D.h("9.7 Software self-tests (SELF-TEST)", 2)
    D.table(["Self-test (python selfcheck.py)", "Result", "Time"], sc,
            f"Automated self-tests, {sc_pass}/{len(sc)} pass (2026-10-14, fakes only)", [8, 3, 3], size=8.5)
    D.p("The machine-cycle test drives the whole trigger > inspection > decision > FIFO > PLC-command loop against "
        "FakeLadder, a scan-by-scan emulation of the decoded ladder, including staggered cameras, association "
        "faults, late rejects and the SQLite record. python selfcheck.py --full adds dataset, training maths, "
        "inference, detector and the whole GUI.", align="j")
    D.h("9.8 Summary of results", 2)
    D.table(["Area", "Result", "Evidence"], [
        ["Control software", f"{sc_pass} / {len(sc)} self-tests pass", "SELF-TEST"],
        ["Classifier (EfficientNet-B0)", f"macro-F1 {b0lf['macro_f1']:.3f}, {b0lf['false_pass']} false PASS, "
                                         f"{b0lf['false_reject']} false REJECT of {b0lf['n_images']}", "HELD-OUT TEST"],
        ["Detector (YOLOv8s v3)", f"mAP50 {v8s['test']['map50']:.3f}, mAP50-95 {v8s['test']['map50_95']:.3f}", "HELD-OUT TEST"],
        ["Detector (YOLOv8n v3)", f"mAP50 {v8n['test']['map50']:.3f}, mAP50-95 {v8n['test']['map50_95']:.3f}", "HELD-OUT TEST"],
        ["Classifiers on the production camera", "33 / 33 frames called defective", "REAL CAMERA"],
        ["Detector on the production camera", "bottles found, empty scenes clean; caps unreliable", "REAL CAMERA"],
        ["Real line", f"{B.finals.get('PASS', 0)} PASS / {B.finals.get('REJECT', 0)} REJECT / "
                      f"{B.finals.get('FAULT', 0)} FAULT of {n_bottles}", "REAL LINE"]], "Results summary", [5, 8, 3])

    # ================================================================ 10 software
    D.h("10. The application", 1)
    D.p(f"One Windows desktop application ({NAME}, CustomTkinter) does everything: label images, train the AI, run "
        "the line, talk to the PLC, show one answer per bottle, keep the production record and manage models.",
        align="j")
    D.table(["Group", "Page", "What it does"], [
        ["PRODUCTION", "Production", "machine state, start checklist, both cameras, current bottle, counters, alarms; "
                                     "START / STOP / RESET FAULT, CAMERA TEST, TEST INSPECTION; engineer row"],
        ["", "History", "per day or shift: counts, yield, defects, bottles/hour, alarms, CSV"],
        ["", "Database", "bottles / alarms / runs tables, filters, CSV for Excel, printable report"],
        ["", "Health", "computer, PLC, cameras, models, storage, alarms; searchable logs"],
        ["DATA", "Label", "thumbnail grid, multi-label tagging, AI pre-labels, undo, import"],
        ["", "Defects", "add / rename / delete defect classes"],
        ["", "Annotate", "Annotation Studio: boxes and polygons, model proposals, review queues, export"],
        ["", "Data health", "class balance, ROI, duplicate / leak warnings"],
        ["MODEL", "Train / Analysis", "train a CANDIDATE; curves and per-defect results"],
        ["", "Models", "registry: Validate, Approve, ACTIVATE, Reject, Roll back"],
        ["ENGINEERING", "Live", "tuning view, one verdict, engineer details, auto-collect"],
        ["", "Machine", "PLC connection (simulator / real), live I/O, armed test PASS / REJECT, simulation check"],
        ["", "Camera", "measures what each camera mode really delivers; focus / exposure lock"],
        ["SYSTEM", "Settings", "text size, theme, evidence policy, engineer PIN, About"]],
        "Application pages (15). Operator mode shows only PRODUCTION; engineer mode shows all", [2.6, 2.8, 10.6], size=8.5)
    D.pic(F20 / "fig23_app_overlay_cam3.jpg", "Inspection overlay produced by the application (TEST INSPECTION)", 11)
    D.h("10.1 Annotation Studio", 2)
    D.pic(FIG / "k10_annotation_review.png", "Annotation review: a model proposes, a person decides", 15)
    D.p("Review states are kept in annotations.json (proposals, rejected, per-box model id and confidence). Every "
        "action saves atomically and is written to annotation_log.jsonl. 389 defect boxes were bootstrapped as "
        "proposals for review (229 label_damaged, 71 label_skewed, 89 cap_tilted) to train a defect-aware detector.",
        align="j")
    D.h("10.2 Alarms, database and logs", 2)
    D.p("25 coded alarms, each with a severity, message and operator action (for example PLC_COMM_FAULT, "
        "CAMERA_DISCONNECTED, UNSURE_RESULT, BOTTLE_UNTRIGGERED, REJECT_DEADLINE_MISSED). RESET FAULT acknowledges "
        "them. SQLite, one file per product (projects/<slug>/production/production.db): runs (job, product, recipe "
        "hash, model ids, settings), inspections (one row per bottle) and alarms. Rows are only appended. Seven "
        "rotating event logs (app, camera, ai, plc, machine, alarm, production): one line per event, never per frame.",
        align="j")
    D.h("10.3 Tools around the application", 2)
    D.bullets(["**System report** (system_report.bat): measures the PC, COM ports, ladder T0/T1 and R1-R13, line "
               "settings, cameras alone and together, model latency, and checks them against a specification.",
               "**Simulation check:** runs the ladder simulator and every self-test side by side. A ladder FAIL means "
               "the ladder or settings are wrong; a self-test FAIL means the code is wrong.",
               "**Model archive** tool; **brand** assets (logo, icon, launcher shortcut)."])

    # ================================================================ 11 timeline
    D.h("11. Work done: timeline", 1)
    D.table(["Date (2026)", "Milestone"], [
        ["Aug 09", "Multi-project design: one folder per product, labels.csv as source of truth"],
        ["Aug-Sep", "Labelling tool, scene-based split, ROI calibration, the five data / skew bugs found and fixed"],
        ["Sep 19", "EfficientNet-B0 checkpoint 20260919-164511 trained (the active classifier)"],
        ["Sep-Oct", "Stage 2 detection dataset (594 images, 1,994 boxes), YOLOv8n training, model benchmark"],
        ["Oct 03", "System audit; ladder decoded from the ISPSoft file; FakeLadder emulation; two-camera USB limit measured"],
        ["Oct 04", "First read-only link to the real PLC (COM5)"],
        ["Oct 05", "Missing-cap root cause found; split v3; background shortcut caught and the candidate rejected; "
                   "model registry and gates; ladder requirement check"],
        ["Oct 06-07", f"First real-line trial ({n_bottles} bottles) recorded in production.db"],
        ["Oct 08", "Retrain of 5 classifier architectures and YOLOv8n / YOLOv8s on v3"],
        ["Oct 13", "Industrial upgrade round: one high-resolution clock, bracketed triggers, fast poll, AI off the "
                   "deadline thread, three-zone decision, threshold / temperature calibration, defect localisation, "
                   "settings audit; black book and final review documents"],
        ["Oct 14", "System report tool, Annotation Studio review workflow, NexaInspect brand, production software plan"],
        ["Oct 15", "This knowledge book"]], "Timeline (from the project documents)", [2.6, 13.4], size=9)
    D.h("11.1 Bugs found and fixed (do not reintroduce)", 2)
    D.table(["Bug", "Effect", "Fix"], [
        ["Random per-block split", "fake 0.000 recall for a rare class", "scene-based split with per-class quota"],
        ["Fixed 25-frame blocks", "fake 1.000 scores", "detect real scene boundaries"],
        ["Train / serve skew", "different crop and JPEG cache shifted scores", "one model_input path, PNG cache"],
        ["ROI not scaled with frame size", "model read background confidently", "store roi_frame and rescale"],
        ["Defect with zero training images could fire", "random alarms", "threshold 1.01 (disabled)"],
        ["Coarse clock (15.6 ms)", "adjacent frames shared a timestamp", "perf_counter clock"],
        ["Empty labels exported for unannotated images", "false negatives in YOLO training", "annotate.exportable"]],
        "Bugs found and fixed", [4.5, 5.5, 6], size=8.5)

    # ================================================================ 12 status
    D.h("12. Status: what works and what does not", 1)
    D.table(["Area", "In software", "On the machine"], [
        ["Per-bottle cycle, FIFO, reject timing, alarms, database", "works, self-tested", "run once (trial); handshake-limited"],
        ["PLC link", "simulator / fake tested", "read-only link + trial run"],
        ["PLC program", "structure correct, timers match settings",
         "no E-stop input, interlock, answer timeout, watchdog or multi-bottle queue"],
        ["Belt speed and distances", "calibration wizard exists", "saved values, no calibration record"],
        ["Two cameras", "stations, association, auto-reconnect", "one USB 2.0 hub: only one 1080p stream"],
        ["Classifier", f"test macro-F1 {b0lf['macro_f1']:.3f}", "fails on production-camera frames"],
        ["Detector", f"test mAP50 {v8s['test']['map50']:.3f}", "finds bottles; caps unreliable"],
        ["Segmentation", "interface only", "no model"]], "Status", [5, 5.2, 5.8], size=8.5)
    D.h("12.1 Open items", 2)
    D.numbered(["Collect production-camera data in the enclosure (target >= 2,000 good and >= 300 per defect), "
                "including real missing-cap bottles; review and retrain; validate through the registry gates.",
                "Ladder changes (owner's decision): bottle FIFO in the PLC so triggers are not masked, E-stop "
                "input, Y0 interlock, PC-answer timeout, heartbeat.",
                "Cameras on separate USB 3 ports; calibrate belt speed and distances on the machine.",
                "LED latency test (trigger -> decision -> command -> cylinder).",
                "Train a defect-aware detector (v4) after the bootstrapped defect boxes are reviewed; detector-guided "
                "classifier crop.",
                "Physical commissioning in the 16-step order; v1.0 is reserved for the release that passes it."])

    # ================================================================ 13 roadmap
    D.h("13. Roadmap", 1)
    D.pic(F21 / "fig_plan_01_roadmap.png", "Overall plan: software phases, freeze gate, hardware, validation, release", 13.5)

    # ================================================================ 14 Q&A
    D.h("14. Questions examiners and reviewers ask", 1)
    qa = [
        ("Why a classifier AND a detector?", "The classifier judges appearance defects (damaged label, water level, "
         "skew) from the whole bottle. The detector finds components, so a missing part can be found as a missing box "
         "in its zone. Each covers what the other cannot."),
        ("Why EfficientNet-B0?", f"Best held-out macro-F1 of five architectures ({b0lf['macro_f1']:.3f}) at about "
         "8 ms per image, small enough for the RTX 3050 next to YOLO."),
        ("Why is validation F1 1.000 but test lower?", "Validation saturates because the dataset has few independent "
         "scenes. That is why the separate test set exists and is quoted; it is read once, never used for choices."),
        ("Why not split randomly?", "The images are consecutive video frames; a random split puts copies of the same "
         "bottle in train and test and gives fake accuracy (bug 1 and 2 in Chapter 11)."),
        ("Why is the input 192 x 448?", "The bottle is tall; a square input wastes half the pixels on padding where "
         "the water line and label print are."),
        ("What happens if a camera fails mid-run?", "Its frames go stale, the bottle becomes FAULT and is rejected, an "
         "alarm is raised and the camera is reopened in the background. Three FAULTs in a row halt the line."),
        ("How do you know when to fire the cylinder without an encoder?", f"Time: travel = distance / speed "
         f"({dist:.0f} mm / {speed:.1f} mm/s = {travel:.1f} s). M1 is sent at trigger + travel - T0 and the PLC "
         "times T0, so the cylinder fires on arrival. A tracking interface is ready for an encoder later."),
        ("What if the decision is late?", "The reject is not fired late (it could hit the wrong bottle). The bottle "
         "is recorded FAULT with an alarm to remove it by hand."),
        ("Why does the PC not drive the cylinder directly?", "The PLC is deterministic and owns safety-relevant "
         "outputs; a PC can freeze. The PC only answers PASS or REJECT."),
        ("Why did the real-line trial fail?", "Two causes, both measured: the PLC handles one bottle at a time "
         "(triggers masked during a reject), and the models were trained on another camera (domain gap). The safety "
         "logic itself behaved correctly."),
        ("How do you stop a bad model reaching production?", "Training only makes a CANDIDATE. Activation needs a "
         "held-out test, real-camera numbers within limits and a shortcut check, then approval; rollback exists."),
        ("What is the white-background shortcut?", "A model trained on missing-cap images that were the only white-"
         "background images learned 'white = missing cap'. A dedicated check now catches this before activation."),
        ("How is false PASS vs false REJECT balanced?", "False PASS is the dangerous error: unknown or unsure is "
         "rejected (three-zone rule). False REJECTs are reduced with more good-bottle data, not with lower thresholds."),
        ("What would make it industrial?", "Production-camera data, a PLC bottle queue, hardware-triggered "
         "global-shutter cameras, an encoder, a reject-confirm sensor, a hardwired safety circuit, and a "
         "validation protocol (IQ/OQ/PQ, repeatability study)."),
        ("How is it tested without the machine?", "Every module has a self-test; FakeLadder emulates the decoded "
         "ladder scan by scan; the ladder simulator runs the real .isp file against 14 scenarios."),
        ("Can it inspect another product?", "Yes: a new project folder with data, a detector trained on its parts and "
         "a recipe (JSON). No code change.")]
    for q, a in qa:
        D.qa(q, a)

    # ================================================================ 15 run
    D.h("15. How to run and test", 1)
    D.table(["Task", "Command"], [
        ["Install and launch (Windows)", "run.bat   (or the NexaInspect desktop shortcut)"],
        ["Quick self-check (~2 min, no window)", "python selfcheck.py"],
        ["Full self-check (incl. GUI)", "python selfcheck.py --full"],
        ["Check the ladder file", "python -m plc.ladder_check   /   python -m plc.ladder_sim"],
        ["System report", "system_report.bat   (add --plc for a read-only PLC test)"],
        ["Train a classifier candidate", "python -m vision.train --epochs 25"],
        ["Score it on the test set", "python -m registry.model_bench cls-test <stamp>"],
        ["Train a detector", "python -m registry.model_bench yolo --model yolov8n.pt --data v3 --workers 0"],
        ["Rebuild this book", "python FINAL_YEAR_BLACKBOOK/22_System_Knowledge_Book/_make_knowledge_figures.py, then "
                              "_build_knowledge_book.py"]], "Commands", [5.5, 10.5], size=8.5)

    # ================================================================ appendices
    D.h("Appendix A: where each chapter goes in the black book", 1)
    D.table(["Black-book chapter", "Take from this book"], [
        ["Abstract / Introduction", "1, 2"], ["Literature review", "9.1 (metrics); black book ch. 2"],
        ["Problem statement, objectives", "2"], ["System requirements", "3 (hardware table), 4"],
        ["System architecture", "4, Figures 'System architecture' and 'Threads'"],
        ["Hardware and PLC", "3, 3.1, 3.2, timing diagram + worked example"],
        ["Dataset", "6 (all figures)"], ["Methodology / model development", "7, 8"],
        ["Implementation", "5, 10"], ["Results", "9"], ["Discussion / limitations", "6.5, 9.5, 9.6, 12"],
        ["Conclusion / future work", "1 (conclusion), 12.1, 13"], ["Viva preparation", "14"]],
        "Mapping to the black book", [6, 10], size=9)
    D.h("Appendix B: key settings", 1)
    D.table(["Setting", "Value now", "Meaning"], [
        ["plc_com", str(SETTINGS.get("plc_com")), "serial port of the real PLC"],
        ["plc_t0_s / plc_t1_s", f"{t0} / {t1}", "must equal the ladder timers"],
        ["conveyor_mm_s", str(speed), "belt speed (0 = not measured)"],
        ["inspection_to_reject_mm", str(dist), "camera to cylinder"],
        ["line_cameras", str(SETTINGS.get("line_cameras")), "camera indices used by the line"],
        ["line_task", str(SETTINGS.get("line_task")), "which AI stages run"],
        ["fault_action", str(SETTINGS.get("fault_action")), "REJECT: a FAULT bottle is never passed"],
        ["evidence_policy", str(SETTINGS.get("evidence_policy", "REJECT_AND_FAULT")), "which bottles keep a picture"],
        ["auto_activate_trained_model", str(SETTINGS.get("auto_activate_trained_model", False)), "keep false"],
        ["active classifier", str(CONFIG.get("active_model")), "projects/om_bottle/config.json"],
        ["detector weights", str(SETTINGS.get("detector_weights", "models/stage2_yolo/stage2_best.pt")), "YOLO in use"]],
        "Settings (settings.json, per PC)", [4.6, 4.4, 7], size=8.5)
    D.h("Appendix C: where things are", 1)
    D.table(["Need", "File"], [
        ["Overview and status", "README.md, docs/roadmap/FEATURE_STATUS.md"],
        ["Everything for a new team member", "docs/TEAM_HANDBOOK.md"],
        ["Rules the code relies on", "CLAUDE.md"],
        ["PLC contract and ladder", "docs/roadmap/PLC_COMMUNICATION.md, docs/hardware/PLC_LADDER_REQUIREMENTS.md"],
        ["Operate and commission", "docs/guides/PRODUCTION_HMI_AND_COMMISSIONING.md"],
        ["Industrial upgrade audit", "INDUSTRIAL_UPGRADE_PLAN.md"],
        ["Black book, final review", "FINAL_YEAR_BLACKBOOK/20_Final_Report/"],
        ["Production software plan", "FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/"],
        ["Retrain results 2026-10-08", "FINAL_YEAR_BLACKBOOK/19_Retrain_2026-10-08/"]], "Document map", [6, 10], size=9)
    D.h("Appendix D: glossary", 1)
    D.table(["Term", "Meaning"], [
        ["PLC / ladder", "industrial controller / its program"], ["Trigger", "bottle at the photo-eye (M2 on)"],
        ["FIFO", "the queue of bottles between camera and cylinder, each with its own deadline"],
        ["FAULT", "the system cannot vouch for this bottle; rejected"],
        ["NOT INSPECTED", "bottle sensed at X0 but no trigger was possible"],
        ["Recipe", "what a complete product looks like: anchor + required parts + zones"],
        ["Scene", "group of near-duplicate frames of one bottle pass, kept whole in a split"],
        ["ROI", "region of interest: the part of the frame the classifier looks at"],
        ["Domain gap", "difference between training images and the production camera's images"],
        ["Hard example", "an image where a person corrected the model's guess"],
        ["Candidate / validated / approved / active", "model lifecycle states"],
        ["Evidence", "the picture kept for a bottle"]], "Glossary", [4.5, 11.5], size=9)
    D.save("NexaInspect_System_Knowledge_Book.docx")


if __name__ == "__main__":
    build()
