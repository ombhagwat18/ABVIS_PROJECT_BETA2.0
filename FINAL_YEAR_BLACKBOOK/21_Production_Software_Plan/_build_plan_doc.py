"""Builds ABVIS_Production_Software_Plan.docx (black-book chapter / annex) from the flowcharts and live project files.

    python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_make_flowcharts.py
    python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_build_plan_doc.py

Read-only for the project: reads settings.json, projects/om_bottle/production/production.db and
data/selfcheck_*.txt; writes only into this folder.
"""
from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREYC = RGBColor(0x55, 0x60, 0x6B)
DATE = "14 October 2026"

# ================================================================== live facts
S = json.loads((ROOT / "settings.json").read_text())
db = sqlite3.connect(ROOT / "projects/om_bottle/production/production.db")
finals = collections.Counter(r[0] for r in db.execute("select final from inspections"))
n_runs = db.execute("select count(*) from runs").fetchone()[0]
n_bottles = sum(finals.values())
sc_file = sorted((HERE / "data").glob("selfcheck_*.txt"))[-1]
sc_lines = [l for l in sc_file.read_text().splitlines() if l.strip()]
sc_total = next((l.split()[0] for l in sc_lines if "passed" in l), "?")
sc_rows = [l.split(None, 1)[1].rsplit(None, 1) + [l.split()[0]] for l in sc_lines if l.startswith(("ok", "FAIL"))]


# ================================================================== docx helpers (same look as 20_Final_Report)
class Doc:
    def __init__(self, title):
        self.d = Document()
        self.fig_n = self.tab_n = 0
        s = self.d.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.left_margin, s.right_margin, s.top_margin, s.bottom_margin = Cm(3.0), Cm(2.0), Cm(2.2), Cm(2.2)
        st = self.d.styles["Normal"]
        st.font.name, st.font.size = "Calibri", Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
            h = self.d.styles[f"Heading {lvl}"]
            h.font.name, h.font.size, h.font.color.rgb, h.font.bold = "Calibri", Pt(size), NAVY, True
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
            h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
            h.paragraph_format.keep_with_next = True
        f = s.footer.paragraphs[0]
        f.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = f.add_run(f"{title}   |   page ")
        r.font.size, r.font.color.rgb = Pt(8.5), GREYC
        self._field(f, "PAGE")

    def _field(self, par, instr):
        r = par.add_run()
        for t in ("begin", "instr", "separate", "end"):
            if t == "instr":
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = instr
            else:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            r._r.append(e)
            if t == "separate":
                t2 = OxmlElement("w:t"); t2.text = "Right-click > Update field"; r._r.append(t2)

    def h(self, text, lvl=1):
        return self.d.add_heading(text, lvl)

    def _runs(self, par, text, bold=False, italic=False, size=None, color=None):
        for i, part in enumerate(text.split("**")):
            if part:
                r = par.add_run(part)
                r.bold, r.italic = bold or i % 2 == 1, italic
                if size:
                    r.font.size = Pt(size)
                if color:
                    r.font.color.rgb = color

    def p(self, text="", bold=False, italic=False, size=None, align=None, color=None):
        par = self.d.add_paragraph()
        self._runs(par, text, bold, italic, size, color)
        if align == "c":
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "j":
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return par

    def bullets(self, items, style="List Bullet"):
        for it in items:
            par = self.d.add_paragraph(style=style)
            self._runs(par, it)
            par.paragraph_format.space_after = Pt(2)

    def numbered(self, items):
        self.bullets(items, "List Number")

    def table(self, header, rows, caption=None, widths=None, size=9):
        if caption:
            self.tab_n += 1
            c = self.d.add_paragraph()
            r = c.add_run(f"Table {self.tab_n}: {caption}")
            r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), NAVY
            c.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htext in enumerate(header):
            cell = t.rows[0].cells[i]; cell.text = ""
            run = cell.paragraphs[0].add_run(str(htext)); run.bold = True; run.font.size = Pt(size)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                self._runs(cells[i].paragraphs[0], str(v), size=size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def fig(self, name, caption, width=16.0):
        self.fig_n += 1
        par = self.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(FIG / name), width=Cm(width))
        par.paragraph_format.keep_with_next = True
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.fig_n}: {caption}")
        r.italic, r.font.size, r.font.color.rgb = True, Pt(9), GREYC

    def note(self, text):
        t = self.d.add_table(rows=1, cols=1)
        t.style = "Light Shading Accent 1"
        cell = t.rows[0].cells[0]; cell.text = ""
        self._runs(cell.paragraphs[0], text, size=9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def brk(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def toc(self):
        self._field(self.d.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')


# ================================================================== content
def build():
    d = Doc("ABVIS - Production-Ready Software Plan")

    # ---------------- title
    for _ in range(5):
        d.p()
    d.p("ABVIS", bold=True, size=30, align="c", color=NAVY)
    d.p("AI-Based Bottle Visual Inspection System", size=15, align="c", color=NAVY)
    d.p()
    d.p("Production-Ready Software Plan and Process Flow", bold=True, size=20, align="c")
    d.p("Black book chapter / annex", italic=True, size=12, align="c", color=GREYC)
    for _ in range(6):
        d.p()
    d.p(f"Prepared {DATE}", align="c", color=GREYC)
    d.p("Status labels used throughout: DONE (software, self-tested with fakes), SOFTWARE (work that needs no "
        "hardware), SIMULATOR (proven against the PLC simulator / ladder emulation), PHYSICAL (proven on the real "
        "machine). Nothing in this document is evidence that a hardware test has passed.",
        italic=True, size=9.5, align="c", color=GREYC)
    d.brk()
    d.h("Contents", 1)
    d.toc()
    d.brk()

    # ---------------- 1
    d.h("1. Purpose and approach", 1)
    d.p("This chapter sets out how the ABVIS software will be completed to a production-ready standard, and how it "
        "will then be proven on the physical inspection machine. It is the working plan for the final phase of the "
        "project and the reference for what “finished” means.", align="j")
    d.p("The approach is **software first, hardware second**. Every part of the system that can be built and proven "
        "without the machine (settings, PLC protocol, AI pipeline, reporting, fault handling) is completed and "
        "verified in software and in the PLC simulator first. The physical commissioning then has one job: to "
        "confirm on the real conveyor what the software has already shown in simulation. This keeps time on the "
        "machine short, keeps people away from moving parts while software is still changing, and makes every "
        "hardware test answer one clear question.", align="j")
    d.note("**Production-ready is declared only after physical proof.** The software can be frozen before the "
           "hardware work, but the system is called production-ready (release v1.0) only after a filmed end-to-end "
           "cycle (one good bottle PASS, one defective bottle REJECT) and after a validated model meets the "
           "acceptance gates in Section 7 on images from the production camera.")
    d.p("The order of work follows the project's guiding principle: **first prove the bottle-inspection machine, "
        "then harden and validate the bottle AI, then generalise the proven components into a universal "
        "inspection platform.**", align="j")

    # ---------------- 2
    d.h("2. Where the project stands today", 1)
    d.p(f"The software core is built. On {DATE} the quick self-check suite passed **{sc_total}** (every check "
        "runs with fake cameras, fake models and a scan-by-scan emulation of the PLC ladder). One real-line trial "
        f"was recorded earlier: **{n_runs} runs, {n_bottles} bottles: {finals.get('PASS', 0)} PASS, "
        f"{finals.get('REJECT', 0)} REJECT, {finals.get('FAULT', 0)} FAULT** (production.db). The trial did what a "
        "trial should: it exposed the two limits that this plan removes.", align="j")
    d.table(["Area", "Status", "Evidence level"], [
        ["Desktop application: 15 pages, operator and engineer modes", "Implemented", "SOFTWARE (GUI self-test)"],
        ["Machine cycle: trigger, per-bottle FIFO, deadlines, one result per bottle", "Implemented",
         "SOFTWARE + ladder emulation"],
        ["Decision engine: recipe, three-zone thresholds, frame vote, camera fusion", "Implemented", "SOFTWARE"],
        ["PLC service: Modbus ASCII, single owner, only M0 / M1 written", "Implemented",
         "SIMULATOR; read-only real link (user-stated)"],
        ["Safety latch, coded alarms, event logs, machine state", "Implemented", "SOFTWARE"],
        ["Production record: SQLite, CSV, evidence images, History / Database pages", "Implemented", "SOFTWARE"],
        ["Model registry: candidate > validated > approved > active, rollback", "Implemented", "SOFTWARE"],
        ["Real-line trial", f"{n_bottles} bottles, {finals.get('FAULT', 0)} FAULT",
         "PHYSICAL (shows the limits below)"],
        ["Classifier and detector accuracy on the production camera", "Not yet adequate",
         "Good on own test sets; fails off-domain"],
        ["Complete physical PASS / REJECT cycle with the cylinder", "Not proven", "-"],
        ["Excel workbook export", "Not implemented", "CSV + printable report exist"],
        ["Segmentation", "Interface only, no trained model", "-"],
    ], "Current status by area", widths=[8.2, 3.6, 4.2])
    d.h("2.1 The two limits found on the real line", 2)
    d.bullets([
        "**Throughput: the PLC handles one bottle at a time.** The ladder has one trigger bit (M2) and one reject "
        "timer pair (T0/T1). While a bottle waits for its answer, or while a reject is in progress, the next bottle's "
        "trigger is masked. Most trial bottles were therefore recorded as FAULT “not inspected”. Software cannot "
        "recover a trigger the PLC never exposes; the fix is a new PLC protocol (Phase B).",
        "**Domain gap: the models were trained on a different camera.** The classifier was trained on a black "
        "backdrop and the detector on webcam-viewer screenshots. On the EMEET production cameras their scores fall "
        "into the uncertain band. The fix is data from the production camera (Phase F), not more training epochs.",
    ])
    d.h("2.2 Configuration found today that must change before production", 2)
    d.table(["Setting (settings.json)", "Value today", "Required for production"], [
        ["fault_action", S.get("fault_action"), "REJECT (a fault must never become a PASS)"],
        ["threshold_shift / threshold_overrides", f"{S.get('threshold_shift')} / "
         f"{len(S.get('threshold_overrides', {}))} overrides", "removed: thresholds come from the activated model"],
        ["decision_rules.det_min_conf_by_class",
         ", ".join(f"{k} {v}" for k, v in S.get("decision_rules", {}).get("det_min_conf_by_class", {}).items()),
         "validated values from the production-camera test set"],
        ["plc_com", S.get("plc_com"), "the real PLC port, pinned by adapter identity"],
        ["conveyor_mm_s / inspection_to_reject_mm", f"{S.get('conveyor_mm_s')} / {S.get('inspection_to_reject_mm')}",
         "measured with the speed calibration wizard (timestamped record)"],
        ["plc_t0_s / plc_t1_s", f"{S.get('plc_t0_s')} / {S.get('plc_t1_s')}",
         "equal to the presets in the program loaded in the PLC"],
    ], "Settings to correct", widths=[5.2, 4.0, 6.8])

    # ---------------- 3
    d.brk()
    d.h("3. Overall plan", 1)
    d.p("Figure 1 shows the plan as a process flow. Phase A comes first because every later phase depends on a "
        "safe, consistent baseline. Phases B, C and D are independent software streams that can run side by side. "
        "A software freeze gate closes the software work; only then does hardware commissioning start. Each gate "
        "has an explicit “No” path: the work loops back rather than moving on with a known problem.", align="j")
    d.fig("fig_plan_01_roadmap.png", "Overall plan: software phases, freeze gate, hardware commissioning, "
          "validation and release", width=14.5)

    # ---------------- 4
    d.brk()
    d.h("4. Software phases (no hardware needed)", 1)
    d.h("Phase A - Safe, consistent software baseline", 2)
    d.table(["#", "Task", "Done when"], [
        ["A1", "Restore safe production settings: fault_action REJECT; remove the live threshold shift, threshold "
               "overrides and lowered detector gates from the production path (allowed only in a logged "
               "commissioning mode).", "Start checklist passes with the model's own thresholds; self-test asserts it"],
        ["A2", "Decide on the 22 white-background missing-cap images (they taught a background shortcut).",
         "missing_cap examples come only from the production setup"],
        ["A3", "Model registry lists models_archive and can roll back to it; Stage 2 data.yaml files use valid "
               "paths.", "Registry self-test covers archive rollback; export validation passes"],
        ["A4", "Pin the PLC serial port by adapter identity; the GUI never silently rewrites plc_com.",
         "Wrong port is refused with a clear message"],
        ["A5", "Full verification (selfcheck --full including the GUI self-test) and status documents corrected "
               "(FEATURE_STATUS, CURRENT_SYSTEM, README, CLAUDE.md).", "All checks green; documents match code"],
    ], "Phase A tasks", widths=[1.0, 9.5, 5.5])
    d.h("Phase B - Line protocol v2 (simulator only)", 2)
    d.p("The present handshake answers one bottle at a time. Protocol v2 lets the PC answer every bottle at once "
        "and lets the PLC keep its own queue of pending rejects and fire the cylinder itself. The ladder belongs to "
        "the PLC owner and is never edited by the software project: both sides are designed together, proven "
        "in the scan simulator, and enabled on the real PLC only after the ladder change is made and tested.",
        align="j")
    d.table(["#", "Task", "Done when"], [
        ["B1", "Write the protocol v2 specification: per-bottle id and verdict, PLC reject queue (shift register or "
               "data-register ring), acknowledgement, fault behaviour; list the rungs in "
               "PLC_LADDER_REQUIREMENTS.md.", "Reviewed by the project team and the PLC owner"],
        ["B2", "Implement v2 in the ladder emulation, the ladder scan simulator, PLCService and MachineCycle behind "
               "a protocol setting; v1 remains the default.", "60 bottles/min in simulation, 0 untriggered bottles; "
                                                               "all v1 tests still pass"],
        ["B3", "Heartbeat watchdog in both directions (PC toggles a bit; the PLC stops or rejects if it stops).",
         "Simulated PC freeze leads to a safe PLC state"],
        ["B4", "Inspection window no longer than bottle pitch / belt speed (about 0.3-0.5 s, 3-5 frames).",
         "No association faults at 100 mm pitch in simulation"],
    ], "Phase B tasks", widths=[1.0, 9.5, 5.5])
    d.h("Phase C - AI pipeline", 2)
    d.table(["#", "Task", "Done when"], [
        ["C1", "Review the 389 bootstrapped defect boxes and draw the remaining region boxes (about 12 hours of "
               "human review).", "Every image accepted by a person"],
        ["C2", "Detector v4 dataset: annotations, split by physical bottle, export, validation.",
         "Export validation passes; no bottle in two splits"],
        ["C3", "Train a YOLOv8s v4 candidate (one training job at a time on the 4 GB GPU).",
         "Candidate registered with provenance; never auto-activated"],
        ["C4", "Detector-guided crop for the classifier through the single shared preprocessing path.",
         "Training and live inference produce identical crops"],
        ["C5", "Target-based thresholds (defect recall >= 99.5 %, good-bottle false alarms <= 1 %) and the "
               "acceptance gates of Section 7 in the registry.", "Gates enforced at validation"],
    ], "Phase C tasks", widths=[1.0, 9.5, 5.5])
    d.p("These models remain candidates: they still carry the domain gap until Phase F adds production-camera data.",
        italic=True, size=10)
    d.h("Phase D - Operations, traceability and reporting", 2)
    d.table(["#", "Task", "Done when"], [
        ["D1", "Excel workbook generated from a consistent SQLite snapshot: Production_Log, PLC_Log, Alarm_Log, "
               "Shift_Report, Model_Performance. Runs downstream; never in the reject timing loop.",
         "Workbook totals equal database totals"],
        ["D2", "Commissioning report export: every test labelled SOFTWARE / SIMULATOR / PHYSICAL with timestamps.",
         "One-click report from the Health page"],
        ["D3", "Photo-eye and camera latency test tool, ready to run on the machine; Start warns without a recent "
               "measurement.", "Tool self-tested with fakes"],
        ["D4", "Recovery tests with fakes: application restart mid-run, PLC disconnect and reconnect, camera "
               "unplug, full disk.", "No bottle silently passes in any case"],
        ["D5", "Operator quick-reference guide for the Production page.", "Reviewed by an operator"],
    ], "Phase D tasks", widths=[1.0, 9.5, 5.5])
    d.h("Software freeze gate", 2)
    d.bullets([
        "selfcheck --full green, including the GUI self-test;",
        "protocol v2 proven in the scan simulator at the target rate, with v1 still passing;",
        "status documents match the code and settings; every claim carries its evidence label.",
    ])

    # ---------------- 5
    d.brk()
    d.h("5. Target per-bottle production process", 1)
    d.p("Figure 2 is the process every bottle follows once the line is running. It is the behaviour the software "
        "already implements and the behaviour the hardware commissioning must confirm. Two properties matter most: "
        "**one inspection id and one final result per physical bottle**, and **no path from a fault to a PASS**.",
        align="j")
    d.fig("fig_plan_02_bottle_cycle.png", "Per-bottle production flow with every fault branch", width=15.0)
    d.table(["Situation", "System response"], [
        ["Readiness checklist fails (PLC, models, cameras, timing, safety settings)", "Start refused, reason shown"],
        ["A camera gives no fresh frame at start", "CAMERA_DISCONNECTED, no start"],
        ["Stale frame, too few bottle frames, model error, uncertain association", "FAULT -> physically rejected"],
        ["Score between the pass and reject limits (unsure)", "FAULT (UNSURE), counted as its own KPI"],
        ["REJECT decided too late to reach the right bottle", "Not fired; FAULT + missed-reject alarm"],
        ["PLC does not acknowledge a command", "FAULT alarm; the command is never retried blindly"],
        ["STOP, E-stop input or 3 FAULTs in a row", "HALT latch: no commands sent, triggers recorded as FAULT"],
        ["X0 rises but the PLC exposes no trigger", "Bottle recorded as NOT INSPECTED (FAULT)"],
    ], "Fault rules (none of them can produce a PASS)", widths=[9.0, 7.0])

    # ---------------- 6
    d.h("6. Hardware commissioning (later)", 1)
    d.p("Hardware work starts after the software freeze. Each stage is read-only or guarded until the stage before "
        "it has passed, and every result is recorded with the label PHYSICAL.", align="j")
    d.table(["Stage", "What is proven", "Acceptance"], [
        ["E0 Inspect and freeze", "Back up configuration and PLC files; compare the program loaded in the PLC with "
                                  "the saved .isp", "Loaded ladder known; differences listed"],
        ["E1 Read-only PLC link", "Real COM port, 9600 7E1 station 1, PLC RUN, X/M/Y/T/C read; photo-eye -> X0 -> M2",
         "Every photo-eye pass seen once; latency recorded"],
        ["E2 Timers", "T0/T1 presets and timer base agreed with measured distance, belt speed and cylinder stroke",
         "Ladder and settings equal; reviewed change"],
        ["E3 Guarded handshake", "Armed M0 (PASS) and M1 (REJECT) with the actuator isolated per the technician's "
                                 "procedure", "M2/M1 clear, counters step, Y0 pulse timed"],
        ["E4 Cameras", "Both EMEETs on the final USB layout, enclosure and lighting; focus, exposure, ROI locked",
         "Resolution, FPS and frame age measured together"],
        ["E5 Timing", "Belt speed (>= 3 runs), photo-eye-to-camera and -to-cylinder distances, cylinder response",
         "Calibration record saved; schedule matches"],
        ["E6 End-to-end", "Good -> PASS; defective -> REJECT; alternating sequence; fault injection; long run",
         "Physical bottles = database = PLC counters"],
    ], "Hardware commissioning stages", widths=[3.3, 7.7, 5.0])
    d.note("Safety: the software HALT latch is not an emergency stop. The hardwired E-stop and guarding are verified "
           "independently, and no reject is fired until the timer behaviour is confirmed. Python never writes the "
           "conveyor or cylinder outputs (Y0/Y1); the PLC owns them.")

    # ---------------- 7
    d.brk()
    d.h("7. Model and data lifecycle, and the acceptance gates", 1)
    d.p("Figure 3 shows how a model reaches production and how production feeds the next model. Training never "
        "activates a model by itself; every activation passes the gates below and is logged with a rollback path.",
        align="j")
    d.fig("fig_plan_03_model_lifecycle.png", "Data and model lifecycle", width=16.0)
    d.table(["Metric", "Gate (production-camera test set, bottles never seen in training)"], [
        ["Defective bottles passed (escape rate)", "<= 0.1 %"],
        ["Good bottles rejected (false reject)", "<= 1 %"],
        ["Unsure rate", "<= 2 %"],
        ["Repeatability", "same bottle x 50 passes -> same verdict >= 49 times"],
        ["Per-defect recall", ">= 99 % for every class with >= 100 test presentations"],
        ["Shortcut check", "no background or lighting shortcut (e.g. white background = defect)"],
    ], "Acceptance gates for model activation", widths=[6.0, 10.0])
    d.p("Data target for Phase F: at least 2,000 good and 300 per defect presentations captured on the final "
        "cameras, enclosure and lighting, split by physical bottle so no bottle appears in two splits.", align="j")

    # ---------------- 8
    d.h("8. Definition of production-ready (release v1.0)", 1)
    d.numbered([
        "Software freeze gate passed (Section 4).",
        "All hardware commissioning stages E0-E6 passed and recorded as PHYSICAL.",
        "Filmed end-to-end cycle: one known-good bottle PASS and one known-defective bottle REJECT by the cylinder.",
        "Active classifier and detector meet the acceptance gates of Section 7 on production-camera data.",
        "A long stability run reconciles physical bottles, database rows, CSV rows and PLC counters exactly.",
        "Every fault case in the fault-rules table (Section 5) demonstrated on the machine without a single silent PASS.",
        "Status documents and this chapter updated with the physical evidence.",
    ])

    # ---------------- 9
    d.h("9. Risks and mitigations", 1)
    d.table(["Risk", "Mitigation"], [
        ["One-bottle PLC handshake limits throughput", "Protocol v2 (Phase B); until then, spaced bottles at low "
                                                        "speed"],
        ["Models fail on the production camera (domain gap)", "Production-camera data and retraining (Phase F)"],
        ["Two webcams share USB bandwidth; rolling shutter on a moving belt",
         "Separate USB controllers, measured modes; industrial cameras after v1.0"],
        ["Program in the PLC differs from the saved file; timer presets unknown",
         "Stage E0 comparison before any reject"],
        ["4 GB GPU and 16 GB RAM limit training", "One training job at a time; data-loader workers limited"],
        ["Python 3.8 is end-of-life", "Upgrade after v1.0, not during commissioning"],
        ["Scope growth before the machine works", "Platform work (Phase G) only after v1.0"],
    ], "Risks", widths=[7.5, 8.5])

    # ---------------- 10
    d.h("10. Decisions needed", 1)
    d.numbered([
        "Approve the Phase A settings changes (fault_action, threshold shift, detector gates).",
        "Remove the white-background missing-cap images from the classifier dataset.",
        "Protocol v2 form (shift register or data-register ring) and who changes the ladder.",
        "Add the openpyxl dependency for the Excel workbook (not installed today).",
        "Target bottle (250 ml or 1 litre) and the defects v1.0 must demonstrate.",
        "Camera hardware budget: keep the webcams for v1.0 or move to industrial cameras.",
    ])

    # ---------------- 11
    d.h("11. Engineering rules that govern every phase", 1)
    d.bullets([
        "One application, one repository; PLCService is the only owner of the PLC link.",
        "The PLC owns the physical outputs; Python never writes Y0 or Y1.",
        "No camera, model, timing or communication fault may become a PASS.",
        "One inspection id and one final result per physical bottle, from fresh frames only.",
        "No model is activated automatically; splits are leakage-safe; screenshot metrics are not production "
        "evidence.",
        "PLC protocol changes are made on both sides together; I/O addresses and timer values are never guessed.",
        "Tests never touch real production data, settings, labels or active models.",
        "Every result states whether it came from software, the simulator or the physical machine.",
    ])

    # ---------------- appendix
    d.h("Appendix A - Software self-check result", 1)
    d.p(f"Command: python selfcheck.py (CPU). Result: **{sc_total}**. All checks use fakes; none is a hardware "
        "test.", size=10)
    d.table(["Check", "Time", "Result"], [[r[0], r[1], r[2].upper()] for r in sc_rows], None, widths=[9, 3, 3],
            size=8.5)
    d.h("Appendix B - Rebuilding this chapter", 1)
    d.p("python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_make_flowcharts.py", size=9.5)
    d.p("python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_build_plan_doc.py", size=9.5)
    d.p("In Word, right-click the table of contents and choose Update Field.", size=9.5, italic=True)

    out = HERE / "ABVIS_Production_Software_Plan.docx"
    d.d.save(out)
    print("wrote", out.name)


if __name__ == "__main__":
    build()
