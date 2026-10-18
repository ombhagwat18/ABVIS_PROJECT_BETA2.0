"""Builds the black book and the final-review / presentation document (.docx) from the figures and the metrics files.

    python FINAL_YEAR_BLACKBOOK/20_Final_Report/_make_figures.py     # figures first
    python FINAL_YEAR_BLACKBOOK/20_Final_Report/_build_docs.py
"""
from __future__ import annotations

import collections
import csv
import json
import sqlite3
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
DATA = HERE / "data"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREYC = RGBColor(0x55, 0x60, 0x6B)

NICE = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet18", "efficientnet_b1": "EfficientNet-B1",
        "mobilenet_v3_small": "MobileNetV3-Small", "convnext_tiny": "ConvNeXt-Tiny"}
CLASSES = ["damaged_bottle", "damaged_label", "missing_cap", "missing_label", "skewed_bottle", "skewed_label",
           "tilt_cap", "water_level"]


def jload(p):
    return json.loads(Path(p).read_text())


# ================================================================== data
cls_runs = []
for d in sorted((ROOT / "projects/om_bottle/models").glob("*/metrics.json")):
    cls_runs.append({"m": jload(d), "t": jload(d.parent / "test_metrics.json")})
ACT = ROOT / "models_archive/previous/om_bottle_models/20260919-164511"
active = {"m": jload(ACT / "metrics.json"), "t": jload(ACT / "test_metrics.json")}
archived = []
for d in sorted((ROOT / "models_archive/previous/om_bottle_models").glob("*/metrics.json")):
    archived.append(jload(d))


def yrows(run):
    rows = list(csv.DictReader(open(ROOT / f"models/stage2_yolo/{run}/results.csv")))
    return [{k.strip(): float(v) for k, v in r.items()} for r in rows]


yolo = {}
for tag, run, cand in (("YOLOv8n v3", "bench_yolov8n_v3", "candidate_yolov8n_v3.json"),
                       ("YOLOv8s v3", "bench_yolov8s_v3", "candidate_yolov8s_v3.json")):
    rows = yrows(run)
    best = max(rows, key=lambda x: 0.1 * x["metrics/mAP50(B)"] + 0.9 * x["metrics/mAP50-95(B)"])
    yolo[tag] = {"rows": rows, "best": best, "c": jload(ROOT / "models/stage2_yolo" / cand)}

lab = list(csv.reader(open(ROOT / "projects/om_bottle/labels.csv")))
lab_h, lab_rev = lab[0], [r for r in lab[1:] if r[-1] == "1"]
counts = {c: sum(1 for r in lab_rev if r[i] == "1") for i, c in enumerate(lab_h[1:-1], 1)}
counts["good"] = sum(1 for r in lab_rev if all(x == "0" for x in r[1:-1]))
man = list(csv.DictReader(open(ROOT / "vision_dataset/manifests/unified_manifest.csv", encoding="utf-8")))
clsman = [r for r in man if r["source_dataset"] == "classification"]
scenes = {c: len({r["scene_id"] for r in clsman if r["cls_class"] == c}) for c in set(r["cls_class"] for r in clsman)}
n_scenes = len({r["scene_id"] for r in clsman})

db = sqlite3.connect(ROOT / "projects/om_bottle/production/production.db")
finals = collections.Counter(r[0] for r in db.execute("select final from inspections"))
reasons = collections.Counter()
for final, data in db.execute("select final, data from inspections"):
    rs = json.loads(data or "{}").get("reason") or ""
    reasons["Not inspected: M2 still ON (previous bottle unanswered)" if "M2 already ON" in rs else
            "Not inspected: M1 reject cycle masks the trigger" if "M1 ON" in rs else
            "Line halted after 3 consecutive FAULTs" if "line halted" in rs else
            "No bottle found by the detector" if "no bottle found" in rs else
            "Stale trigger at line start" if "stale trigger" in rs else
            "REJECT (several defects reported)" if final == "REJECT" else "Camera fault / other"] += 1
real = jload(DATA / "real_camera_model_check.json")
SELFCHECK = [("Decision engine", 2.3), ("Machine cycle", 16.6), ("Time tracking", 0.1), ("Machine state", 0.1),
             ("Alarms", 0.1), ("Production record", 2.1), ("Event logs", 0.1), ("Model registry", 0.3),
             ("PLC protocol", 2.3), ("PLC service", 11.0), ("PLC commissioning window", 4.3),
             ("Ladder decoder", 0.1), ("Ladder simulator", 3.6), ("Active learning", 2.0), ("Stable verdict", 0.1),
             ("Production export", 2.1), ("Threshold calibration", 0.2), ("Model activation gates", 1.0)]


# ================================================================== docx helpers
class Doc:
    def __init__(self, title):
        self.d = Document()
        self.fig_n = 0
        self.tab_n = 0
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
        self._footer(title)

    def _field(self, par, instr):
        r = par.add_run()
        for t, v in (("begin", None), ("instr", instr), ("separate", None), ("end", None)):
            if t == "instr":
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = v
            else:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            r._r.append(e)
            if t == "separate":
                r2 = OxmlElement("w:t"); r2.text = "Right-click > Update field"; r._r.append(r2)

    def _footer(self, title):
        f = self.d.sections[0].footer.paragraphs[0]
        f.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = f.add_run(f"{title}   |   page ")
        run.font.size, run.font.color.rgb = Pt(8.5), GREYC
        self._field(f, "PAGE")

    def h(self, text, lvl=1):
        return self.d.add_heading(text, lvl)

    def p(self, text="", bold=False, italic=False, size=None, align=None, color=None):
        par = self.d.add_paragraph()
        self._runs(par, text, bold, italic, size, color)
        if align == "c":
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "j":
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return par

    def _runs(self, par, text, bold=False, italic=False, size=None, color=None):
        # **bold** segments inside text
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            r = par.add_run(part)
            r.bold = bold or (i % 2 == 1)
            r.italic = italic
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = color

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
            cell = t.rows[0].cells[i]
            cell.text = ""
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
        return t

    def fig(self, name, caption, width=16.0):
        path = FIG / name
        if not path.exists():
            self.p(f"[missing figure {name}]", italic=True)
            return
        self.fig_n += 1
        par = self.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(path), width=Cm(width))
        par.paragraph_format.keep_with_next = True
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.fig_n}: {caption}"); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = GREYC

    def note(self, text):
        t = self.d.add_table(rows=1, cols=1)
        t.style = "Light Shading Accent 1"
        cell = t.rows[0].cells[0]; cell.text = ""
        self._runs(cell.paragraphs[0], text, size=9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def toc(self):
        par = self.d.add_paragraph()
        self._field(par, 'TOC \\o "1-3" \\h \\z \\u')

    def brk(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, name):
        self.d.save(HERE / name)
        print("wrote", name)


# ================================================================== shared tables
def t_cls_test(doc, cap):
    rows = []
    for r in cls_runs + [active]:
        m, t = r["m"], r["t"]
        lf, fu = t["test_leak_free"], t["test"]
        rows.append([m["stamp"] + (" (active)" if r is active else ""), NICE[m["arch"]], f"{fu['macro_f1']:.3f}",
                     f"{lf['macro_f1']:.3f}", f"{lf['macro_precision']:.3f}", f"{lf['macro_recall']:.3f}",
                     f"{lf['pass_reject_accuracy']:.3f}", lf["false_pass"], lf["false_reject"],
                     f"{t['latency']['ms_mean']:.1f}" if t.get("latency") else "-"])
    doc.table(["Checkpoint", "Architecture", "F1 full", "F1 leak-free", "Precision", "Recall", "PASS/REJECT acc",
               "False PASS", "False REJECT", "ms/img"], rows, cap, size=8)


def t_perclass(doc, cap):
    b0 = next(r for r in cls_runs if r["m"]["arch"] == "efficientnet_b0")
    pd = b0["t"]["test_leak_free"]["per_defect"]
    rows = []
    for c in CLASSES:
        x = pd.get(c, {})
        if not x.get("n_pos"):
            rows.append([c, 0, x.get("threshold", "-"), "-", "-", "-", "-", "-", "no test positives"])
        else:
            rows.append([c, x["n_pos"], x["threshold"], f"{x['precision']:.3f}", f"{x['recall']:.3f}", f"{x['f1']:.3f}",
                         x["tp"], x["fp"], x["fn"]])
    doc.table(["Defect", "Test positives", "Threshold", "Precision", "Recall", "F1", "TP", "FP", "FN"], rows, cap, size=8.5)


def t_epochs(doc, cap):
    rows = []
    for r in cls_runs:
        m = r["m"]
        secs = sum(h.get("seconds", 0) for h in m["history"])
        rows.append([m["stamp"], NICE[m["arch"]], f"{m['epochs_run']} / {m['epochs']}", m["best_epoch"],
                     m["n_train"], m["n_val"], m["batch"], m["lr"], f"{secs / 60:.1f} min", m["device"]])
    for tag, y in yolo.items():
        c = y["c"]
        rows.append([tag, "YOLOv8 (Ultralytics 8.1.0)", f"{len(y['rows'])} / {c['train_args']['epochs']}",
                     int(y["best"]["epoch"]), "453", "89", c["train_args"]["batch"], "auto (SGD/AdamW)",
                     f"{c['train_minutes']} min", "cuda"])
    doc.table(["Model", "Architecture", "Epochs run / max", "Kept epoch", "Train img", "Val img", "Batch", "LR",
               "Time", "Device"], rows, cap, size=8)


def t_archived(doc, cap):
    rows = []
    for m in archived:
        secs = sum(h.get("seconds", 0) for h in m.get("history", []))
        rows.append([m["stamp"], NICE.get(m["arch"], m["arch"]), f"{m.get('epochs_run', len(m['history']))} / {m['epochs']}",
                     m.get("best_epoch") or "not recorded", m["n_train"], f"{secs / 60:.1f} min", m["device"]])
    doc.table(["Checkpoint", "Architecture", "Epochs", "Kept epoch", "Train img", "Time", "Device"], rows, cap, size=8.5)


def t_yolo(doc, cap):
    rows = []
    for tag, y in yolo.items():
        c = y["c"]
        rows.append([tag, f"{c['n_params'] / 1e6:.1f} M", f"{c['size_mb']} MB", f"{c['val']['map50']:.3f}",
                     f"{c['val']['map50_95']:.3f}", f"{c['test']['map50']:.3f}", f"{c['test']['map50_95']:.3f}",
                     f"{c['test']['precision']:.3f}", f"{c['test']['recall']:.3f}",
                     f"{c.get('benchmark_gpu', {}).get('latency_ms_mean', 0):.1f}"])
    doc.table(["Candidate", "Params", "Size", "Val mAP50", "Val mAP50-95", "Test mAP50", "Test mAP50-95",
               "Test P", "Test R", "GPU ms"], rows, cap, size=8)


def t_yolo_perclass(doc, cap):
    rows = []
    for tag, y in yolo.items():
        for cl, v in y["c"]["test"]["per_class"].items():
            rows.append([tag, cl, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['map50']:.3f}", f"{v['map50_95']:.3f}"])
    doc.table(["Candidate", "Class", "Precision", "Recall", "mAP50", "mAP50-95"], rows, cap, size=8.5)


def t_dataset(doc, cap):
    order = sorted(counts, key=lambda c: -counts[c])
    rows = [[c, counts[c], scenes.get(c, "white bg, 1 session" if c == "missing_cap" else "-")] for c in order]
    rows.append(["**Total reviewed**", f"**{len(lab_rev)}**", f"**~{n_scenes}**"])
    doc.table(["Class", "Reviewed images", "Distinct scenes"], rows, cap, size=9)


def t_real(doc, cap):
    rows = []
    for st, v in real["classifier"].items():
        n = len(v["real"])
        rows.append([st, NICE[v["arch"]], f"{sum(1 for x in v['real'] if x['hits'])} / {n}",
                     f"{sum(len(x['hits']) for x in v['real']) / n:.1f}"])
    doc.table(["Classifier", "Architecture", "Real frames called defective", "Defects per frame"], rows, cap, size=9)


def t_production(doc, cap):
    rows = [[k, v, f"{100 * v / sum(reasons.values()):.0f} %"] for k, v in reasons.most_common()]
    doc.table(["Outcome / reason", "Bottles", "Share"], rows, cap, size=9)


def t_selfcheck(doc, cap):
    rows = [[n, "PASS", f"{s:.1f} s"] for n, s in SELFCHECK]
    doc.table(["Self-test (python selfcheck.py)", "Result", "Time"], rows, cap, size=8.5)


# ================================================================== BLACK BOOK
def blackbook():
    D = Doc("ABVIS - Automated Bottle Visual Inspection System")
    # ---------------- title page
    for _ in range(3):
        D.p()
    D.p("AUTOMATED BOTTLE VISUAL INSPECTION SYSTEM (ABVIS)", bold=True, size=22, align="c", color=NAVY)
    D.p("AI-based defect detection and PLC-controlled rejection for a bottle conveyor", italic=True, size=13, align="c")
    D.p()
    D.p("A project report submitted in partial fulfilment of the requirements for the degree of", align="c")
    D.p("Bachelor of Technology in Robotics & Automation Engineering", bold=True, size=13, align="c")
    D.p()
    D.p("Submitted by", align="c")
    D.p("[Student name 1 - roll no.]\n[Student name 2 - roll no.]\n[Student name 3 - roll no.]", align="c")
    D.p()
    D.p("Under the guidance of", align="c")
    D.p("[Guide name, designation]", align="c")
    D.p()
    D.p("[Department of Robotics & Automation Engineering]\n[College / Institute name, City]\n[University name]",
        align="c")
    D.p("Academic year 2026-27", bold=True, align="c")
    D.brk()
    # ---------------- certificate / declaration / acknowledgement
    D.h("Certificate", 1)
    D.p("This is to certify that the project entitled \"Automated Bottle Visual Inspection System (ABVIS)\" is a "
        "bona fide work carried out by [student names] in partial fulfilment of the requirements for the award of the "
        "degree of Bachelor of Technology in Robotics & Automation Engineering of [University name] during the academic "
        "year 2026-27.", align="j")
    D.p("\n\n[Guide]                                [Head of Department]                                [External examiner]")
    D.h("Declaration", 1)
    D.p("We declare that this report is our own work, that the results reported are those produced by the system "
        "described, and that every result obtained with simulated or fake hardware is labelled as such.", align="j")
    D.h("Acknowledgement", 1)
    D.p("[Thank the guide, the department, the laboratory staff and anyone who supplied hardware or bottles.]", italic=True)
    D.brk()
    # ---------------- abstract
    D.h("Abstract", 1)
    D.p("Manual inspection of filled bottles is slow and inconsistent. ABVIS is a machine-vision inspection station "
        "for a 250 ml water-bottle conveyor. A photo-eye triggers a Delta DVP-SS2 PLC, the PLC signals a Windows PC "
        "over Modbus ASCII, and the PC captures frames from two USB cameras taken strictly after the trigger. Two AI "
        "models examine them: a multi-label EfficientNet classifier that scores eight defect classes, and a YOLOv8 "
        "detector that locates the bottle, cap and label. A rule-based decision engine votes the evidence over several "
        "frames and cameras into one PASS, REJECT or FAULT per bottle. The PC answers the PLC with a PASS or REJECT "
        "bit timed so that the PLC fires the pneumatic reject cylinder when that bottle arrives. Every bottle is "
        "recorded in an SQLite production database.", align="j")
    b0 = next(r for r in cls_runs if r["m"]["arch"] == "efficientnet_b0")
    D.p(f"On held-out test data the best classifier (EfficientNet-B0) reached a macro-F1 of "
        f"{b0['t']['test_leak_free']['macro_f1']:.3f} with zero defective bottles passed, and the YOLOv8s detector "
        f"reached a test mAP50 of {yolo['YOLOv8s v3']['c']['test']['map50']:.3f}. The control software passes 18 of 18 "
        f"automated self-tests against a simulated PLC. A first real-line trial (200 bottles) and a test on real "
        f"production-camera frames exposed what laboratory scores hide: a domain gap between the training images and "
        f"the production camera, decision thresholds that collapse to 0.05, and a one-bottle PLC handshake that left "
        f"78 % of bottles uninspected. The report documents the system, the results, these failures, and an "
        f"industrial upgrade plan.", align="j")
    D.p("**Keywords:** machine vision, quality inspection, EfficientNet, YOLOv8, multi-label classification, PLC, "
        "Modbus, conveyor reject system.")
    D.brk()
    D.h("Table of Contents", 1)
    D.toc()
    D.p("(In Word: right-click the table above and choose Update Field to fill in page numbers.)", italic=True, size=9)
    D.brk()

    # ================= 1 Introduction
    D.h("1. Introduction", 1)
    D.h("1.1 Background", 2)
    D.p("Bottling lines check every container for a missing or tilted cap, a damaged or skewed label, a damaged or "
        "skewed bottle, and a wrong fill level. Human inspectors tire, disagree and cannot keep up with line speed. "
        "Machine vision replaces them with cameras, lighting and software that decide on every unit and remove the "
        "bad ones automatically.", align="j")
    D.h("1.2 What ABVIS is", 2)
    D.p("ABVIS is one Windows desktop application plus a PLC program. The application labels images, trains the AI, "
        "manages model versions, runs the inspection line, talks to the PLC, raises alarms and keeps the production "
        "record. The PLC owns the conveyor, the reject delay and the reject cylinder. The first product is the 250 ml "
        "Bisleri water bottle.", align="j")
    D.fig("fig01_architecture.png", "ABVIS system architecture")
    D.h("1.3 Design principle: unknown is never good", 2)
    D.p("A broken camera, a crashed model, a stale frame or a bottle no camera saw becomes FAULT, and a FAULT bottle "
        "is rejected by default. The PC never switches an output directly: it only writes a PASS (M0) or REJECT (M1) "
        "bit, and the PLC does the rest.", align="j")
    D.h("1.4 Organisation of the report", 2)
    D.bullets(["Chapter 2 reviews the techniques used.", "Chapters 3-4: problem, objectives and requirements.",
               "Chapters 5-6: architecture, hardware and PLC.", "Chapter 7: datasets.",
               "Chapter 8: AI model development.", "Chapter 9: decision engine and machine cycle.",
               "Chapter 10: software implementation.", "Chapter 11: testing and results.",
               "Chapters 12-15: discussion, limitations, conclusion and future work."])

    # ================= 2 Literature
    D.h("2. Literature Review and Technical Background", 1)
    D.h("2.1 Convolutional classifiers", 2)
    D.p("Residual networks (ResNet) made very deep CNNs trainable through skip connections [2]. EfficientNet scales "
        "depth, width and input resolution together, giving higher accuracy per FLOP [1]. MobileNetV3 targets "
        "low-latency devices [3]. ConvNeXt modernises the plain CNN with transformer-era design choices [4]. All "
        "five are available in torchvision with ImageNet-pretrained weights [10], so a small in-domain dataset can be "
        "used for fine-tuning rather than training from scratch.", align="j")
    D.h("2.2 Multi-label classification", 2)
    D.p("Defects are not mutually exclusive: a bottle can be skewed and have a damaged label. Each defect therefore "
        "gets its own sigmoid output and binary cross-entropy loss, with a per-class positive weight to counter class "
        "imbalance. Each output needs its own decision threshold.", align="j")
    D.h("2.3 Object detection", 2)
    D.p("YOLO detectors predict boxes and classes in one forward pass [5]. Ultralytics YOLOv8 [6] is used here in its "
        "nano (3.0 M parameters) and small (11.1 M) sizes. Detection quality is reported as mean average precision at "
        "IoU 0.5 (mAP50) and averaged over IoU 0.5-0.95 (mAP50-95), as in COCO [13].", align="j")
    D.h("2.4 Training and calibration", 2)
    D.p("Models are optimised with AdamW [11] and a cosine learning-rate schedule [12]. Raw network scores are often "
        "over-confident. Temperature scaling [7] calibrates them so that a 0.9 means about 90 % in practice. This "
        "matters for an industrial accept/reject threshold.", align="j")
    D.h("2.5 PLC and industrial communication", 2)
    D.p("The Delta DVP-SS2 is programmed in ladder logic (IEC 61131-3 [14]) with ISPSoft [9]. The PC exchanges bits "
        "with it over Modbus [8] (ASCII framing, LRC check, RS-232). Safety functions such as the emergency stop must "
        "be hardwired and follow ISO 13849-1 [15]. They must not depend on PC software.", align="j")

    # ================= 3 Problem & objectives
    D.h("3. Problem Statement and Objectives", 1)
    D.h("3.1 Problem statement", 2)
    D.p("Design and build an automated station that inspects every bottle on a conveyor, detects cap, label, body and "
        "fill-level defects, removes defective bottles with a pneumatic cylinder at the right moment, never passes a "
        "bottle it could not inspect, and keeps a traceable record of every decision.", align="j")
    D.h("3.2 Objectives", 2)
    D.numbered(["Build a labelled multi-label dataset of bottle defects with a leak-free, scene-based split.",
                "Train and compare several CNN classifiers and select one on a held-out test set.",
                "Build and train a component detector (bottle, cap, label) for presence checks such as a missing cap.",
                "Build a decision engine that turns multi-frame, multi-camera evidence into PASS / REJECT / FAULT.",
                "Integrate the PLC: trigger, PASS/REJECT handshake and time-based reject scheduling without an encoder.",
                "Provide an operator HMI, alarms, model registry and a production database.",
                "Verify everything with automated self-tests, then run a first trial on the real line and report it honestly."])
    D.h("3.3 Scope", 2)
    D.p("In scope: one product (250 ml bottle), two cameras, one reject station, time-based tracking. Out of scope "
        "for this stage: encoder tracking, multi-product changeover, segmentation-based label measurement and "
        "certified safety functions.", align="j")

    # ================= 4 Requirements
    D.h("4. System Requirements", 1)
    D.h("4.1 Hardware", 2)
    D.table(["Component", "Specification"], [
        ["PC", "12th Gen Intel Core i7-12700H (14 cores / 20 threads), 16 GB RAM, Windows 11"],
        ["GPU", "NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM"],
        ["Cameras", "2 x EMEET SmartCam Nova 4K (USB, UVC), 1920x1080 MJPG, mounted rotated 90 deg"],
        ["PLC", "Delta DVP-SS2 series, programmed in ISPSoft; RS-232 COM port, Modbus ASCII 9600 7E1, station 1"],
        ["Sensor", "Photoelectric bottle sensor on X0; start / stop push buttons on X1 / X2"],
        ["Actuator", "Festo DSNU pneumatic cylinder with 5/2 solenoid valve on Y0; conveyor motor on Y1"],
        ["Conveyor", "Measured belt speed 108.2 mm/s; photo-eye to reject cylinder 1028 mm"]],
        "Hardware used", [4, 12])
    D.h("4.2 Software", 2)
    D.table(["Package", "Version", "Use"], [
        ["Python", "3.8.0", "application runtime"], ["PyTorch / torchvision", "2.4.1+cu124 / 0.19.1+cu124", "classifier"],
        ["Ultralytics", "8.1.0", "YOLOv8 detector"], ["OpenCV", "4.13.0", "camera capture, image processing"],
        ["CustomTkinter", "6.0.0", "desktop HMI"], ["Pillow / NumPy", "10.4.0 / 1.24.3", "images, arrays"],
        ["pyserial", "3.5", "real PLC link"], ["psutil", "7.2.2", "system monitoring"],
        ["SQLite", "built-in", "production database"], ["ISPSoft", "Delta", "PLC ladder programming / simulator"]],
        "Software stack", [4.5, 5, 6.5])
    D.h("4.3 Functional requirements", 2)
    D.bullets(["Label images, manage defect classes and import datasets.", "Train, evaluate and version AI models.",
               "Capture frames after each PLC trigger from one or more cameras.",
               "Decide PASS / REJECT / FAULT per bottle and answer the PLC once per trigger.",
               "Schedule the reject so the cylinder fires when the bottle reaches it; never fire a late reject.",
               "Show one stable verdict per bottle to the operator; hide engineering detail in operator mode.",
               "Record every bottle, alarm and run; export CSV and reports."])
    D.h("4.4 Non-functional requirements", 2)
    D.bullets(["Safety: unknown is FAULT; FAULT is rejected; the PC never writes PLC outputs.",
               "Timing: the verdict must be ready before trigger + travel - T0.",
               "Traceability: every decision carries the model ids, timings and evidence image.",
               "Robustness: a camera, model or link failure produces FAULT and an alarm, not a crash."])

    # ================= 5 Architecture
    D.h("5. System Architecture", 1)
    D.p("The software is a single Python process organised in packages. Figure 1 shows the components; Table 5 lists "
        "the packages and their size.", align="j")
    D.table(["Package", "Lines", "Responsibility"], [
        ["vision/", "6,609", "dataset, training, inference (Camera, Model), YOLO detector, calibration, verdict, annotation"],
        ["line/", "3,852", "decision engine, machine cycle, tracking, machine state, alarms, production store, logs"],
        ["registry/", "991", "model registry, benchmarking, activation gates"],
        ["ui/", "9,241", "main window, HMI pages, theme, charts, annotation studio"],
        ["plc/", "3,871", "Modbus protocol, address map, client, PLC service, ladder decoder and simulator, tests"],
        ["tools/", "402", "self-check runner, migration, camera planner"],
        ["stage2_dataset/", "2,557", "detection dataset preparation, export, validation"]],
        "Code organisation (about 28,000 lines of Python)", [3.2, 1.6, 11.2])
    D.h("5.1 Data flow for one bottle", 2)
    D.fig("fig02_machine_cycle.png", "Per-bottle machine cycle")
    D.h("5.2 Threads and timing", 2)
    D.bullets(["**Tk main thread:** all widgets; background work reports through a queue drained every 60 ms.",
               "**Camera grab thread (one per camera):** reads frames, stamps them, keeps the last 12.",
               "**Camera score thread:** live-view scoring only (about 15 Hz).",
               "**PLC service thread:** the only owner of the serial / TCP link; polls M2/X0, writes M0/M1.",
               "**Machine-cycle thread:** collects frames per bottle, runs the AI, decides and keeps the FIFO deadlines.",
               "**Evidence writer:** saves images without delaying any PLC command."])
    D.h("5.3 Machine states", 2)
    D.p("One function (line/machine_state.py) derives the machine state for every screen: OFFLINE, NOT_READY, READY, "
        "INITIALIZING, RUNNING, INSPECTING, STOPPING, FAULT, E_STOP and COMMUNICATION_FAULT. START is enabled only in "
        "READY, when the PLC is connected and in RUN, cameras and models are present, timing is valid and the E-stop "
        "is released.", align="j")

    # ================= 6 Hardware & PLC
    D.h("6. Hardware and PLC Integration", 1)
    D.h("6.1 PLC I/O map", 2)
    D.table(["Device", "Meaning", "Who writes it"], [
        ["X0", "Photo-eye: bottle at the inspection station", "field (read-only)"],
        ["X1 / X2", "Start / stop push buttons (SET / RESET Y1)", "field (read-only)"],
        ["M2", "PLC to PC: inspection trigger (bottle ready)", "PLC ladder"],
        ["M0", "PC to PLC: PASS command (PLC self-clears)", "PC - the only way to pass"],
        ["M1", "PC to PLC: REJECT command (held through the reject cycle)", "PC - the only way to reject"],
        ["T0", "Delay from M1 to the cylinder firing", "PLC timer"],
        ["T1", "Reject pulse length", "PLC timer"],
        ["Y0", "Reject solenoid valve / cylinder", "PLC only - the PC never writes Y"],
        ["Y1", "Conveyor motor", "PLC only"],
        ["C0 / C1", "PASS / REJECT counters", "PLC"]], "PLC device map (plc/address_map.py)", [2.2, 9, 4.8])
    D.h("6.2 Communication", 2)
    D.p("plc/protocol.py builds Modbus ASCII frames with an LRC checksum. plc/client.py offers a TCP transport (the "
        "ISPSoft simulator on 127.0.0.1:10002) and a serial transport (the real PLC). plc/service.py is the single "
        "owner of the link. It enforces the command rules: one answer per trigger, the trigger marked answered before "
        "the write, no retries, preconditions (M2 = 1, M0 = M1 = 0) re-read just before writing, and success = the "
        "PLC clearing the command bit.", align="j")
    D.h("6.3 Reject timing without an encoder", 2)
    D.p("travel time = distance (camera to cylinder) / belt speed = 1028 mm / 108.2 mm/s = 9.5 s. The PC sends M1 at "
        "trigger + travel - T0, and the PLC fires Y0 after T0, so the cylinder fires at trigger + travel. A REJECT "
        "that would be later than reject_tolerance_s (0.3 s) is not fired. It is answered with M0, recorded as FAULT, "
        "and an alarm tells the operator to remove the bottle by hand.", align="j")
    D.h("6.4 Ladder verification", 2)
    D.p("plc/ladder_check.py decodes the ISPSoft project (a binary header followed by a raw deflate stream) and checks "
        "13 requirements. plc/ladder_sim.py executes the decoded networks in a scan simulator against 14 scenarios. "
        "Result: the structure (trigger, PASS, REJECT, single Y0 pulse, conveyor start/stop) is correct. The timers "
        "differ from the intended contract, and the ladder lacks an E-stop input, a PC-answer timeout and a "
        "multi-bottle queue.", align="j")

    # ================= 7 Dataset
    D.h("7. Dataset and Data Preparation", 1)
    D.h("7.1 Classification dataset", 2)
    D.p(f"Project om_bottle holds {len(lab) - 1} images, {len(lab_rev)} of them reviewed. labels.csv is the single "
        f"source of truth (one row per image, one column per defect). The images are consecutive frames of about "
        f"{n_scenes} physical bottle passes ('scenes'), so the real sample size is the number of scenes.", align="j")
    t_dataset(D, "Classification dataset per class")
    D.fig("fig04_dataset_distribution.png", "Images and distinct scenes per class")
    D.fig("fig05_dataset_samples.png", "Example training image for each class")
    D.h("7.2 Leak-free splitting", 2)
    D.p("A random split would put near-identical frames of the same bottle in both training and test and report false "
        "accuracy. Frames are grouped into scenes by image similarity, and whole scenes go to train, validation or "
        "test (about 70 / 15 / 15). The test set is scored once, after every training decision. A stricter "
        "'leak-free' test subset removes images that a cross-folder hash audit linked to training scenes.", align="j")
    D.h("7.3 Pre-processing", 2)
    D.p("A region of interest (ROI) is measured once from Sobel edge energy, because the clear bottle is nearly as dark "
        "as the backdrop. The crop is letterboxed and centre-cropped to a tall 192 x 448 input that keeps the meniscus "
        "and label detail. Training and inference share one crop function (dataset.model_input), and the crop cache "
        "is lossless PNG (JPEG shifted water-level scores by ~0.15). Augmentation: ±5 deg rotation, ±3 % scale, "
        "brightness/contrast. No horizontal flip, because it would mirror label text and invert skew.", align="j")
    D.h("7.4 Detection dataset", 2)
    D.p("stage2_dataset holds 594 reviewed images with 1,994 boxes of three classes (bottle, cap, label). Split v3 "
        "(scene-based) has 453 train, 89 validation and 52 test images. The v3 split moved the only full-bottle "
        "missing-cap scene into training, so missing-cap detection cannot be validated on the v3 test set.", align="j")
    D.h("7.5 Dataset limitations found", 2)
    D.bullets(["Only 72 good images from 8 scenes; missing_label from 2 scenes.",
               "No image carries more than one defect, although the model is multi-label.",
               "The 22 missing-cap images are the only white-background images, which invites a background shortcut.",
               "All images are 2537 x 1927 from another camera and set-up; the production camera differs (Figure 6)."])
    D.fig("fig06_domain_gap.png", "Training images vs frames from the production camera")

    # ================= 8 AI models
    D.h("8. AI Model Development", 1)
    D.h("8.1 Stage 1: multi-label defect classifier", 2)
    D.p("Backbone pretrained on ImageNet, final layer replaced by one linear output per defect, sigmoid activation. "
        "Loss: BCEWithLogitsLoss with pos_weight = clip(negatives / positives, 1, 50) per class. Optimiser AdamW "
        "(lr 3e-4, weight decay 1e-4), cosine schedule, mixed precision on GPU, batch 32, 25 epochs. Good images and "
        "operator-corrected 'hard examples' are oversampled by a weighted sampler (x5 and x3). The epoch with the best "
        "validation macro-F1 is kept, and each class's threshold is chosen on validation.", align="j")
    D.h("8.2 Training runs", 2)
    t_epochs(D, "Training of the current models (2026-10-08)")
    D.fig("fig07_cls_training_curves.png", "Classifier training curves")
    D.fig("fig12_epochs.png", "Epochs run and epoch kept for every current model")
    D.p("Validation macro-F1 reaches 1.000 by epoch ~12 for every architecture, so the validation set cannot separate "
        "epochs or thresholds. Extra epochs on this data only memorise the training scenes further.", align="j")
    t_archived(D, "Earlier classifier checkpoints (archived)")
    D.h("8.3 Stage 2: component detector", 2)
    D.p("YOLOv8n and YOLOv8s were fine-tuned from COCO weights on split v3: 640 px input, batch 8, at most 80 epochs, "
        "early stop after 20 epochs without improvement, mosaic, HSV and horizontal-flip augmentation, and workers = 0 "
        "(more workers exhausted the 16 GB machine). The best epoch is chosen on validation, and test is evaluated "
        "once. YOLOv8s was still improving at epoch 80 (best epoch 70), so it is under-trained.", align="j")
    D.fig("fig14_yolo_curves.png", "Detector training curves")
    D.fig("fig16b_yolo_results.png", "Ultralytics training summary for YOLOv8s v3", 15)
    D.h("8.4 Stage 3: segmentation", 2)
    D.p("A segmentation interface exists (vision/segment.py) and 546 label polygons are seeded, but none is reviewed "
        "and no segmentation model is trained. Any recipe that requires segmentation therefore returns FAULT.",
        align="j")
    D.h("8.5 Model lifecycle", 2)
    D.p("Training never puts a model into production. A new model is a CANDIDATE. It becomes VALIDATED only with a "
        "held-out test result, a written real-camera check and a background-shortcut check; then APPROVED, then "
        "ACTIVE. The previous model is ARCHIVED and can be rolled back. Gates: good bottles rejected ≤ 2 %, defective "
        "bottles passed ≤ 1 %, at least 30 + 30 real-camera bottles.", align="j")

    # ================= 9 Decision & machine cycle
    D.h("9. Decision Engine and Machine Cycle", 1)
    D.h("9.1 Per-frame findings", 2)
    D.bullets(["**Classifier:** a defect is found when its score ≥ its threshold; empty or NaN scores are FAULT.",
               "**Detector + recipe:** one bottle is chosen (upright aspect ≥ 1.3, area ≥ 2 %, nearest the station "
               "centre). A required part (cap near the top, label in the middle) that is not found in its zone becomes "
               "missing_cap / missing_label. A part outside its zone becomes cap_misplaced.",
               "**No bottle in a frame:** that frame does not vote; no bottle in any frame is FAULT."])
    D.h("9.2 Vote and fusion", 2)
    D.p("Per camera, a defect must appear in a majority of the bottle's frames. Cameras are fused FAULT > REJECT > PASS, "
        "so a bottle passes only if every camera saw it and found it good.", align="j")
    D.h("9.3 Safety latch and alarms", 2)
    D.p("STOP, an E-stop status input, three consecutive FAULT bottles or a crash latch the line. While latched no M0/M1 "
        "is sent and every trigger becomes a FAULT 'not answered' bottle. Coded alarms carry a severity, a message and "
        "an operator action, and RESET FAULT acknowledges them. The hardware E-stop must cut power independently.",
        align="j")
    D.h("9.4 Camera layer and timestamps", 2)
    D.p("Each frame is stamped with time.monotonic() as soon as cap.read() returns and given a per-session sequence "
        "number. The Inspector uses only frames stamped strictly after the bottle's window opens, and stops at the "
        "next bottle's window. Frames or scores older than 1.0 s are FAULT. The analysis in Chapter 11 shows that the "
        "stamp is consistent but not exact.", align="j")
    D.fig("fig22_timestamp_budget.png", "Timestamp error budget of the camera / trigger chain")

    # ================= 10 Implementation
    D.h("10. Software Implementation", 1)
    D.h("10.1 The application", 2)
    D.table(["Group", "Pages", "Purpose"], [
        ["PRODUCTION", "Production, History, Database, Health", "operator screens: run the line, shifts, records, system health"],
        ["DATA", "Label, Defects, Annotate, Data health", "labelling, defect classes, boxes and proposals, dataset checks"],
        ["MODEL", "Train, Analysis, Models", "training, evaluation, model registry"],
        ["ENGINEERING", "Live, Machine, Camera", "tuning view, PLC connection and test, camera benchmark"],
        ["SYSTEM", "Settings", "text size, theme, evidence policy, engineer PIN"]], "Application pages (15)", [3, 5.5, 7.5])
    D.p("Operator mode shows only the PRODUCTION group and one stable verdict per bottle (GOOD, DEFECT: <name>, FAULT). "
        "Engineer mode shows every page and the commissioning tools.", align="j")
    D.fig("fig23_app_overlay_cam3.jpg", "Inspection overlay produced by the application (TEST INSPECTION, camera 3)", 12)
    D.h("10.2 Production database", 2)
    D.p("SQLite, one file per product (projects/<slug>/production/production.db). Tables: runs (job, product, model "
        "ids, recipe, settings), inspections (one row per bottle: result, defects, confidence, PLC status, timings, "
        "evidence path) and alarms. Rows are only appended. CSV and printable reports are exported from the Database "
        "page.", align="j")
    D.h("10.3 Logging", 2)
    D.p("Seven rotating event logs (app, camera, ai, plc, machine, alarm, production): one line per event, never one per "
        "frame. They are searchable from the Health page.", align="j")

    # ================= 11 Results
    D.h("11. Testing and Results", 1)
    D.note("Every result in this chapter states its evidence: SELF-TEST (software with a fake PLC / cameras / models), "
           "HELD-OUT TEST (images never used for training or selection), REAL CAMERA (frames from the production "
           "cameras) or REAL LINE (the physical conveyor and PLC).")
    D.h("11.1 Software self-tests (SELF-TEST)", 2)
    t_selfcheck(D, "Automated self-tests: 18 of 18 pass (CPU, 2026-10-13)")
    D.p("The machine-cycle test drives the full trigger, inspection, decision, FIFO and PLC-command loop against a "
        "scan-by-scan emulation of the decoded ladder, including staggered camera stations, association faults, late "
        "rejects and the SQLite record.", align="j")
    D.h("11.2 Classifier results (HELD-OUT TEST)", 2)
    t_cls_test(D, "Classifier held-out test results")
    D.fig("fig08_cls_test_f1.png", "Classifier test macro-F1")
    D.fig("fig09_cls_perclass_heatmap.png", "Per-class F1 on the leak-free test set")
    t_perclass(D, "EfficientNet-B0 (20261008-081732) per-class results, leak-free test set")
    D.fig("fig10_cls_errors.png", "Bottle-level errors on the leak-free test set")
    D.fig("fig11_cls_speed_accuracy.png", "Accuracy vs inference time and model size", 12)
    D.p("EfficientNet-B0 is the best classifier: it has the highest macro-F1 and no defective bottle passed. tilt_cap "
        "is the weakest class for every architecture (F1 ≈ 0.73). A global image score is a poor measure of a small "
        "angular change; a geometric measurement is better (Chapter 15).", align="j")
    D.fig("fig13_thresholds.png", "Decision thresholds chosen by training vs the proposed three-zone limits")
    D.p("Many thresholds are 0.05. Validation is perfectly separable, and the threshold search returns the first, i.e. "
        "lowest, of the tied best values. Thresholds this low make the classifier fire on any unfamiliar image "
        "(Section 11.5).", align="j")
    D.h("11.3 Detector results (HELD-OUT TEST)", 2)
    t_yolo(D, "Detector candidates, validation and test")
    t_yolo_perclass(D, "Detector per-class test results")
    D.fig("fig15_yolo_perclass.png", "Detector per-class test metrics")
    D.fig("fig16_yolo_pr_curve.png", "Precision-recall curve, YOLOv8s v3, test split", 12)
    D.fig("fig17_yolo_confusion.png", "Normalised confusion matrix, YOLOv8s v3, test split", 12)
    D.h("11.4 Missing-cap test", 2)
    D.p("Detectors v1 and v2 called the green tamper ring a cap (~0.70 confidence) and missed all 12 full-bottle "
        "missing-cap images. v3 trained on that scene, so its result on them is not proof. A classifier trained on "
        "the white-background missing-cap images learned 'white background = missing cap' (25 of 25 capped "
        "white-background bottles flagged) and was rejected.", align="j")
    D.fig("fig25_missing_cap_test.jpg", "Missing-cap test image with detector output (v1)", 11)
    D.h("11.5 Real production-camera frames (REAL CAMERA)", 2)
    D.p("Every model was run on 33 frames from the EMEET production cameras: bottles on the conveyor, the empty belt, "
        "an empty desk and a cable close-up. None shows a defective bottle.", align="j")
    t_real(D, "Classifiers on real production-camera frames")
    D.fig("fig19_real_camera_classifiers.png", "Classifiers on real production-camera frames")
    D.fig("fig20_real_camera_detectors.png", "Detectors on real production-camera frames")
    D.fig("fig18_real_camera_detections.jpg", "YOLOv8s v3 detections on all 33 real frames", 16)
    D.p("Every classifier calls every real frame defective, including empty scenes (EfficientNet-B0 reports "
        "missing_cap on all of them). YOLOv8s v3 behaves best: no boxes on empty scenes and the bottle found at "
        "0.82-0.89. But it often misses the cap (cut off at the top edge of camera 2's view), draws a 0.92 'cap' on "
        "a red cable, and puts low-confidence 'bottle' boxes on the metal guide rail.", align="j")
    D.h("11.6 First real-line trial (REAL LINE)", 2)
    t_production(D, "Real-line trial, 2026-10-06/07 (production.db, 200 bottles)")
    D.fig("fig21_production_trial.png", "Real-line trial outcomes")
    D.fig("fig03_throughput_budget.png", "Time budget of the one-bottle PLC handshake")
    D.p("167 of 200 bottles were never inspected: the ladder raises a new trigger only after the previous bottle is "
        "answered, and a REJECT holds M1 for T0 + T1. With a 3.9 s watch window, a 4.5 s reject dispatch and "
        "T0 + T1 = 7.5 s, the station handles at most ~14 bottles/min if all are good and ~5/min with rejects, "
        "against one bottle every ~0.9 s on the belt. The safety design worked as intended: every uninspected "
        "bottle was recorded as FAULT, and the latch halted the line after three consecutive FAULTs.", align="j")
    D.h("11.7 Summary of results", 2)
    D.table(["Area", "Result", "Evidence"], [
        ["Control software", "18 / 18 self-tests pass", "SELF-TEST"],
        ["Best classifier (EfficientNet-B0)", f"macro-F1 {b0['t']['test_leak_free']['macro_f1']:.3f}, 0 false PASS, "
                                              f"{b0['t']['test_leak_free']['false_reject']} false REJECT of 226", "HELD-OUT TEST"],
        ["Best detector (YOLOv8s v3)", f"mAP50 {yolo['YOLOv8s v3']['c']['test']['map50']:.3f}, "
                                       f"mAP50-95 {yolo['YOLOv8s v3']['c']['test']['map50_95']:.3f}", "HELD-OUT TEST"],
        ["Classifiers on the production camera", "33 / 33 frames called defective", "REAL CAMERA"],
        ["Detector on the production camera", "bottles found, empty scenes clean; caps unreliable", "REAL CAMERA"],
        ["Real line", "0 PASS / 4 REJECT / 196 FAULT; 167 not inspected", "REAL LINE"]], "Results summary", [5, 8, 3])

    # ================= 12 Discussion
    D.h("12. Discussion and Failure Analysis", 1)
    D.table(["Failure", "Root cause", "Remedy"], [
        ["Classifier flags every real frame", "domain gap (camera, background, framing) + thresholds at 0.05 + fixed ROI "
         "from a landscape frame applied to a portrait frame", "production-camera data; detector-guided crop; calibrated, "
         "target-based thresholds"],
        ["Validation saturates at 1.000", "near-duplicate frames, too few scenes", "split by physical bottle; production-camera validation"],
        ["tilt_cap weakest class", "small angle, global score", "angle measurement from oriented boxes / keypoints"],
        ["Detector misses caps / false caps", "few production frames; cap clipped at frame edge", "re-aim camera; add production data and negatives"],
        ["78 % uninspected", "one-bottle handshake + long window + reject dispatch delay", "PLC-side bottle FIFO; window ≤ 0.5 s"],
        ["Timestamps not exact", "15.6 ms clock; stamp after read(); trigger after serial reply",
         "perf_counter clock, bracketed stamps, LED latency test, hardware-triggered cameras"],
        ["fault_action set to PASS", "commissioning setting left on", "REJECT; START refuses PASS outside monitor-only mode"]],
        "Failure analysis", [3.8, 6.4, 5.8], size=8.5)
    D.p("Lesson: laboratory metrics (F1 0.95, mAP 0.98) measure how well a model fits its own image distribution. They "
        "say nothing about a new camera. The decisive tests were the real-camera frames and the real-line trial. The "
        "safety architecture (FAULT by default, the halt latch, no late rejects, a full record) behaved correctly "
        "even though the models did not.", align="j")

    # ================= 13 Limitations
    D.h("13. Limitations", 1)
    D.bullets(["The models are not validated on the production camera; the classifier is not usable on it today.",
               "The dataset is small in independent bottles (~111 scenes) and single-label only.",
               "Missing-cap and missing-label detection are not validated.",
               "Webcams are rolling-shutter, autofocus devices, and two share one USB 2.0 hub (one 1080p stream).",
               "Timing is time-based (no encoder); camera and trigger latencies are not yet measured.",
               "The PLC ladder handles one bottle at a time and lacks an E-stop input and a PC-answer timeout.",
               "Python 3.8 is past end of life; the inspection core runs inside the desktop UI process."])

    # ================= 14 Conclusion
    D.h("14. Conclusion", 1)
    D.p("ABVIS delivers a complete inspection-station software stack: dataset tooling, two trained AI models with "
        "honest held-out evaluation, a model registry with activation gates, a decision engine, a deadline-driven "
        "machine cycle, a PLC link with strict command rules, an operator HMI, alarms and a traceable production "
        "database. All 18 self-tests pass against a simulated PLC. On held-out data the classifier reaches macro-F1 "
        f"{b0['t']['test_leak_free']['macro_f1']:.3f} and the detector mAP50 "
        f"{yolo['YOLOv8s v3']['c']['test']['map50']:.3f}.", align="j")
    D.p("The first real-line trial and the real-camera test show that the system is not yet ready for production. The "
        "models must be retrained on production-camera images, the decision thresholds must become calibrated and "
        "confidence-based, and the PLC must track several bottles at once. The project identified each of these "
        "precisely, with evidence, and Chapter 15 sets out the plan.", align="j")

    # ================= 15 Future work
    D.h("15. Future Work: Industrial Upgrade Plan", 1)
    D.table(["Phase", "Work", "Done when"], [
        ["0 - Safe and consistent", "fault_action REJECT; restore or activate models; remove live threshold shift; fix "
         "launcher; archive old models by dataset version", "START checklist green with safe settings"],
        ["1 - Exact timing, confident decisions", "perf_counter clock; bracketed frame/trigger stamps; fast trigger poll; "
         "AI off the deadline thread; LED latency test; three-zone decision with calibrated scores; production-camera "
         "data collection", "trigger latency p99 < 60 ms; ≥ 2,000 good + ≥ 300 per defect"],
        ["2 - Throughput and localisation", "PLC bottle FIFO; window ≤ 0.5 s; detector v4 with defect classes "
         "(cap_tilted, neck_bare, label_damaged, label_skewed, bottle_damaged, fill_line); detector-guided classifier "
         "crop; YOLOv8s 200 epochs at 960 px", "60 bottles/min in simulation; escape rate ≤ 0.1 % on production test"],
        ["3 - Industrial hardware", "global-shutter hardware-triggered cameras, lighting enclosure, encoder, reject-confirm "
         "sensor, E-stop input, Modbus TCP, heartbeat watchdog", "latency and gates re-measured on new hardware"],
        ["4 - Measurement and validation", "angle and fill-level measurement (OBB / pose), IQ/OQ/PQ, repeatability study, "
         "shift-start challenge bottles, Windows service, Python upgrade", "commercial acceptance"]],
        "Upgrade phases (full detail: INDUSTRIAL_UPGRADE_PLAN.md)", [3.4, 8.6, 4], size=8.5)
    D.h("15.1 Three-zone confidence decision", 2)
    D.p("Each defect gets two limits on its calibrated score. Above reject_min (≈ 0.90) it is a confident defect. Below "
        "pass_max (≈ 0.10) it is confidently absent. In between, the bottle is UNSURE and is rejected and counted "
        "separately. A bottle passes only if every defect is confidently absent in ≥ 80 % of its frames and the "
        "detector sees the bottle (≥ 0.80) and its parts (≥ 0.70). Acceptance targets: escape rate ≤ 0.1 %, false "
        "reject ≤ 1 %, unsure ≤ 2 %.", align="j")
    D.h("15.2 Defect localisation", 2)
    D.p("New detector classes mark where each defect is. Annotations are bootstrapped from the existing component boxes "
        "and image labels (e.g. the cap box of a tilt_cap image becomes cap_tilted) and then reviewed by a person. "
        "Fill level and tilt become measurements in millimetres and degrees, which a QA engineer can tune and audit.",
        align="j")

    # ================= references
    D.h("References", 1)
    D.numbered([
        "M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" ICML, 2019.",
        "K. He, X. Zhang, S. Ren and J. Sun, \"Deep Residual Learning for Image Recognition,\" CVPR, 2016.",
        "A. Howard et al., \"Searching for MobileNetV3,\" ICCV, 2019.",
        "Z. Liu et al., \"A ConvNet for the 2020s,\" CVPR, 2022.",
        "J. Redmon, S. Divvala, R. Girshick and A. Farhadi, \"You Only Look Once: Unified, Real-Time Object Detection,\" CVPR, 2016.",
        "G. Jocher, A. Chaurasia and J. Qiu, \"Ultralytics YOLOv8,\" 2023. https://github.com/ultralytics/ultralytics",
        "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" ICML, 2017.",
        "Modbus Organization, \"MODBUS Application Protocol Specification V1.1b3,\" 2012.",
        "Delta Electronics, \"DVP-ES2/EX2/SS2/SA2/SX2/SE/TP Operation Manual - Programming.\"",
        "J. Deng et al., \"ImageNet: A Large-Scale Hierarchical Image Database,\" CVPR, 2009.",
        "I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization,\" ICLR, 2019.",
        "I. Loshchilov and F. Hutter, \"SGDR: Stochastic Gradient Descent with Warm Restarts,\" ICLR, 2017.",
        "T.-Y. Lin et al., \"Microsoft COCO: Common Objects in Context,\" ECCV, 2014.",
        "IEC 61131-3, \"Programmable controllers - Part 3: Programming languages.\"",
        "ISO 13849-1, \"Safety of machinery - Safety-related parts of control systems.\""])

    # ================= appendices
    D.h("Appendix A: Key settings", 1)
    D.table(["Setting", "Value in trial", "Meaning / recommended"], [
        ["inspection_to_reject_mm / conveyor_mm_s", "1028 / 108.2", "travel 9.5 s (measured)"],
        ["plc_t0_s / plc_t1_s", "5.0 / 2.5", "must equal the ladder's T0 / T1"],
        ["inspect_frames / inspect_window_s", "13 / 3.9", "recommend 3-5 frames in ≤ 0.5 s"],
        ["fault_action", "PASS", "must be REJECT"],
        ["threshold_shift", "-0.3", "remove (thresholds come from the activated model)"],
        ["decision_rules.det_min_conf_by_class", "bottle 0.2, cap 0.2, label 0.4", "bottle 0.80, cap 0.70, label 0.70"],
        ["line_cameras / rotate", "[2, 3] / 90 deg", "lock focus, exposure and white balance as well"]],
        "Configuration used in the trial", [5.5, 4, 6.5], size=8.5)
    D.h("Appendix B: How to run", 1)
    D.bullets(["pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
               "pip install -r requirements.txt ultralytics==8.1.0 pyserial comtypes psutil",
               "python -m vision.calibrate   (once per product)",
               "python gui.py   (the application)",
               "python selfcheck.py [--full]   (all self-tests)",
               "python -m vision.train --epochs 25 --arch efficientnet_b0   (train a candidate)",
               "python -m registry.model_bench yolo --model yolov8s.pt --data v3 --workers 0   (train a detector)"])
    D.h("Appendix C: Glossary", 1)
    D.table(["Term", "Meaning"], [
        ["Scene", "one physical bottle pass recorded as consecutive frames"],
        ["Macro-F1", "unweighted mean of per-class F1 (harmonic mean of precision and recall)"],
        ["mAP50 / mAP50-95", "mean average precision at IoU 0.5 / averaged over IoU 0.5-0.95"],
        ["False PASS (escape)", "defective bottle passed: the error that reaches the customer"],
        ["False REJECT", "good bottle rejected: costs product, not quality"],
        ["FAULT", "the system cannot vouch for the bottle; rejected by default"],
        ["Recipe", "the definition of a complete product: anchor (bottle) and required parts in zones"],
        ["Candidate", "a trained model not yet approved for production"]], "Glossary", [4, 12])
    D.h("Appendix D: Figure and data sources", 1)
    D.p("All figures are generated by FINAL_YEAR_BLACKBOOK/20_Final_Report/_make_figures.py from the metrics files, "
        "the production database and data/real_camera_model_check.json (produced by data/real_camera_model_check.py "
        "on 2026-10-13). YOLO plots come from Ultralytics (19_Retrain_2026-10-08). Screenshots of the application "
        "pages should be added to 13_Screenshots/ before submission.", align="j")
    D.save("ABVIS_Black_Book.docx")


# ================================================================== REVIEW + PRESENTATION
def review():
    b0 = next(r for r in cls_runs if r["m"]["arch"] == "efficientnet_b0")
    lf = b0["t"]["test_leak_free"]
    ys = yolo["YOLOv8s v3"]["c"]
    D = Doc("ABVIS - Final Review and Presentation")
    D.p("ABVIS - Automated Bottle Visual Inspection System", bold=True, size=20, align="c", color=NAVY)
    D.p("Final project review pack: review summary, presentation slides, demo script and viva preparation",
        italic=True, align="c")
    D.p("[Students]  |  [Guide]  |  [Department, College]  |  2026-27", align="c", color=GREYC)
    D.toc()
    D.brk()

    # ------------- part A review summary
    D.h("Part A - Final Review Summary", 1)
    D.h("A.1 One-paragraph summary", 2)
    D.p("ABVIS is an AI machine-vision station that inspects 250 ml bottles on a conveyor and makes a PLC reject the bad "
        "ones. It combines an EfficientNet defect classifier, a YOLOv8 component detector, a voting decision engine and "
        "a Delta PLC handshake over Modbus. It also includes an operator HMI, a model registry and a production "
        f"database. On held-out data the classifier scores macro-F1 {lf['macro_f1']:.3f} (0 defective bottles passed) "
        f"and the detector mAP50 {ys['test']['map50']:.3f}, and the control software passes 18/18 self-tests. A real "
        "trial and a real-camera test showed a camera domain gap, collapsed thresholds and a PLC throughput limit. All "
        "three are diagnosed, with a phased plan to industrial grade.", align="j")
    D.h("A.2 Objectives vs achievement", 2)
    D.table(["Objective", "Status", "Evidence"], [
        ["Leak-free multi-label dataset", "Done (limited size)", f"{len(lab_rev)} reviewed images, ~{n_scenes} scenes, scene split"],
        ["Compare classifiers, select on test", "Done", "5 architectures; EfficientNet-B0 selected"],
        ["Component detector", "Done", f"YOLOv8s v3 test mAP50 {ys['test']['map50']:.3f}"],
        ["Decision engine (vote, fusion, FAULT)", "Done", "self-tests"],
        ["PLC integration and reject timing", "Done in simulation; read-only on the real PLC; trial run", "ladder simulator S1-S14, trial"],
        ["HMI, alarms, registry, database", "Done", "15 pages, SQLite record of 200 bottles"],
        ["Validated on the real line", "Not yet", "trial: 0 PASS / 4 REJECT / 196 FAULT"]], "Objectives", [5, 4.5, 6.5])
    D.h("A.3 Key numbers", 2)
    D.table(["Metric", "Value"], [
        ["Classifier test macro-F1 (EfficientNet-B0, leak-free)", f"{lf['macro_f1']:.3f}"],
        ["Classifier false PASS / false REJECT (226 test bottles)", f"{lf['false_pass']} / {lf['false_reject']}"],
        ["Classifier inference", f"{b0['t']['latency']['ms_mean']:.1f} ms/image (RTX 3050)"],
        ["Detector test mAP50 / mAP50-95 (YOLOv8s v3)", f"{ys['test']['map50']:.3f} / {ys['test']['map50_95']:.3f}"],
        ["Detector inference", f"{ys['benchmark_gpu']['latency_ms_mean']:.1f} ms/image"],
        ["Software self-tests", "18 / 18 pass"],
        ["Code base", "~28,000 lines Python, 7 packages"],
        ["Real-camera frames called defective by classifiers", "33 / 33 (every model)"],
        ["Real-line trial", "200 bottles: 0 PASS, 4 REJECT, 196 FAULT (167 not inspected)"]], "Key numbers", [9, 7])
    D.h("A.4 Contributions", 2)
    D.bullets(["A complete, safety-first inspection architecture: unknown is FAULT, no late rejects, a single PLC owner.",
               "Honest evaluation: scene-based leak-free splits, a test set used once, real-camera and real-line evidence.",
               "A model registry with activation gates and rollback.",
               "PLC ladder decoding and scan simulation to test the PLC program against the software contract.",
               "A diagnosis of why laboratory accuracy did not transfer, with a measurable upgrade plan."])

    # ------------- part B slides
    D.brk()
    D.h("Part B - Presentation (slide by slide)", 1)
    D.p("20 slides, about 15-18 minutes. Each slide lists its title, on-screen points, the figure to show and what to "
        "say.", italic=True)
    slides = [
        ("Title", ["ABVIS - Automated Bottle Visual Inspection System", "AI defect detection + PLC-controlled rejection",
                   "[Students] - [Guide] - [Department, College]"], None,
         "Introduce the team and the one-line idea: a camera inspects every bottle and a cylinder removes the bad ones."),
        ("Problem", ["Manual inspection is slow, tiring and inconsistent",
                     "Defects: missing / tilted cap, damaged / skewed / missing label, damaged / skewed bottle, fill level",
                     "Goal: inspect every bottle, reject bad ones, never pass an uninspected bottle"], "fig05_dataset_samples.png",
         "Show the eight defect classes on real bottles."),
        ("Objectives", ["Dataset with leak-free split", "Classifier + detector", "Decision engine",
                        "PLC integration and reject timing", "HMI, registry, database", "Validate on the real line"], None,
         "These are the seven objectives; the review summary table shows which are done."),
        ("System architecture", ["PC: HMI, cameras, AI, decision, machine cycle, PLC service, database",
                                 "PLC: owns conveyor, T0 delay, T1 pulse, cylinder", "Modbus ASCII over RS-232"],
         "fig01_architecture.png", "Walk left to right: cameras > AI > decision > PLC > cylinder. Stress: the PC never writes an output."),
        ("Machine cycle", ["X0 > M2 trigger > frames after trigger > AI > vote", "PASS: M0 now",
                           "REJECT: M1 at trigger + travel - T0 > PLC fires Y0"], "fig02_machine_cycle.png",
         "Explain time-based tracking: travel 9.5 s from the measured distance and belt speed."),
        ("Hardware", ["Delta DVP-SS2 PLC, photo-eye, Festo cylinder + 5/2 valve", "2 x EMEET Nova 4K cameras",
                      "i7-12700H, RTX 3050 4 GB"], "fig06_domain_gap.png", "Show the real station frames next to the training images."),
        ("Dataset", [f"{len(lab_rev)} reviewed images, ~{n_scenes} scenes", "Scene-based split: no near-duplicates across sets",
                     "Detection: 594 images, 1,994 boxes"], "fig04_dataset_distribution.png",
         "Explain why scenes, not images, are the real sample size."),
        ("Classifier", ["EfficientNet-B0 + 4 other backbones, multi-label sigmoid", "BCE + pos_weight, AdamW, cosine, 25 epochs",
                        "Weighted sampling for good bottles and hard examples"], "fig07_cls_training_curves.png",
         "Point out that validation saturates at 1.0 early: the data, not the epochs, is the limit."),
        ("Classifier results", [f"Best: EfficientNet-B0, test macro-F1 {lf['macro_f1']:.3f}", f"0 defective bottles passed, "
                                f"{lf['false_reject']} good rejected (226)", "Weakest class: tilt_cap"],
         "fig08_cls_test_f1.png", "Compare architectures; justify B0 by accuracy, zero escapes and speed."),
        ("Per-class view", ["Most classes ≈ 1.0 F1", "tilt_cap ≈ 0.73 for every model", "missing_label: no test positives"],
         "fig09_cls_perclass_heatmap.png", "A global score is weak for small angles; measurement is the fix."),
        ("Detector", ["YOLOv8n / YOLOv8s on bottle, cap, label", f"YOLOv8s test mAP50 {ys['test']['map50']:.3f}",
                      "YOLOv8s still improving at epoch 80"], "fig14_yolo_curves.png", "Explain mAP50 and mAP50-95; label is the hardest class."),
        ("Detector quality", ["Per-class precision / recall / mAP", "PR curve and confusion matrix"], "fig15_yolo_perclass.png",
         "Highlight the label mAP50-95 (0.62): box edges of a wrap-around label are ambiguous."),
        ("Decision engine and safety", ["Recipe: required parts in zones", "Majority vote over frames; FAULT > REJECT > PASS",
                                        "Unknown is FAULT; FAULT rejected; late reject never fired; halt latch"],
         None, "This is what keeps the line safe even when the AI is wrong."),
        ("Software", ["15-page HMI, operator / engineer modes", "Model registry: candidate > validated > approved > active",
                      "SQLite production record, alarms, logs", "18 / 18 self-tests"], "fig23_app_overlay_cam3.jpg",
         "Show a screenshot or the live demo."),
        ("Real-camera test", ["Classifiers: 33 / 33 real frames called defective", "Detector: bottles found, empty scenes clean, caps unreliable"],
         "fig19_real_camera_classifiers.png", "The domain gap: lab accuracy did not transfer to the new camera."),
        ("Real-line trial", ["200 bottles: 0 PASS, 4 REJECT, 196 FAULT", "167 not inspected: one-bottle PLC handshake",
                             "Safety worked: everything uninspected recorded as FAULT"], "fig21_production_trial.png",
         "Present honestly: the failure is understood and measurable."),
        ("Root causes", ["Domain gap + thresholds at 0.05", "One-bottle handshake + long window", "Timestamps not exact (15.6 ms clock, serial latency)"],
         "fig03_throughput_budget.png", "Walk through the time budget bar chart."),
        ("Upgrade plan", ["Phase 0 safe settings", "Phase 1 exact timing + three-zone confidence + real data",
                          "Phase 2 PLC FIFO + defect localisation", "Phase 3 industrial cameras, encoder", "Phase 4 measurement, validation"],
         "fig13_thresholds.png", "Explain the three-zone decision: confident defect / confident good / unsure > reject."),
        ("Conclusion", ["Complete, safety-first inspection stack built and tested", "Strong held-out results",
                        "Real-world gaps found with evidence and a plan to close them"], None,
         "End on what was learned: real validation decides, not lab scores."),
        ("Questions", ["Thank you"], None, "Have backup slides ready: dataset split, PLC I/O map, timing formula."),
    ]
    for i, (title, pts, fig, say) in enumerate(slides, 1):
        D.h(f"Slide {i}: {title}", 2)
        D.bullets(pts)
        if fig:
            D.fig(fig, f"Slide {i} visual", 13)
        D.p(f"**Speaker notes:** {say}", size=9.5, color=GREYC)

    # ------------- part C demo
    D.brk()
    D.h("Part C - Live Demo Script (no hardware needed)", 1)
    D.numbered(["Launch: python gui.py. The app opens in OPERATOR mode on the Production page.",
                "Switch to ENGINEER mode (header button). Show the page groups.",
                "Machine page: connect to the ISPSoft simulator (TCP 127.0.0.1:10002), or explain the real COM link.",
                "Production > engineer row > Simulation check: run 'Check the LADDER' (14 scenarios) and 'Check the SOFTWARE' (self-tests).",
                "Production > TEST INSPECTION on a saved image: show the verdict and the overlay.",
                "Label page: show labels.csv-backed labelling and the inbox. Annotate page: show boxes and proposals.",
                "Models page: show the CANDIDATE > ACTIVE lifecycle and gates.",
                "Database page: open the trial records, export the CSV report.",
                "Terminal backup: python selfcheck.py (18/18 pass in about 2 minutes)."])

    # ------------- part D viva
    D.h("Part D - Viva Questions and Answers", 1)
    qa = [
        ("Why multi-label instead of multi-class?", "Defects can co-occur (a skewed bottle with a damaged label). One "
         "sigmoid per defect allows any combination; softmax would force exactly one."),
        ("Why split by scene?", "Images are consecutive video frames. A random split puts near-duplicates of one bottle "
         "in train and test and inflates accuracy. Whole scenes go to one split only."),
        ("Why did you choose EfficientNet-B0?", f"Highest held-out macro-F1 ({lf['macro_f1']:.3f}), zero defective "
         f"bottles passed, and {b0['t']['latency']['ms_mean']:.1f} ms per image."),
        ("What is macro-F1 and why not accuracy?", "The mean of per-class F1. Accuracy is misleading under imbalance: "
         "always answering 'no defect' scores high accuracy while missing every defect."),
        ("Why both a classifier and a detector?", "The classifier judges appearance (damage, skew, fill). The detector "
         "judges presence and position of parts (a missing cap is a bottle without a cap box)."),
        ("What is mAP50?", "Mean average precision when a predicted box counts as correct at IoU ≥ 0.5 with the true box; "
         "mAP50-95 averages IoU thresholds 0.5 to 0.95 and rewards tight boxes."),
        ("How is the reject timed without an encoder?", "Travel time = distance / measured belt speed. The PC sends M1 at "
         "trigger + travel - T0; the PLC waits T0 and fires the cylinder exactly when the bottle arrives."),
        ("What happens if the camera or model fails?", "The bottle becomes FAULT and is rejected by default; three FAULTs "
         "in a row halt the line; an alarm explains why."),
        ("Why does the PC never write PLC outputs?", "Outputs are owned by the deterministic PLC. The PC only answers PASS "
         "or REJECT, so a PC crash cannot fire or block the cylinder directly."),
        ("Why did the classifier fail on the real camera?", "Domain gap: different camera, background, framing and crop; and "
         "thresholds at 0.05 that fire on any unfamiliar image."),
        ("Why were 78 % of bottles uninspected?", "The ladder accepts a new trigger only after the previous bottle is "
         "answered; the window and reject cycle made each bottle occupy the PLC for 4-12 s while bottles arrive every ~0.9 s."),
        ("Would more epochs help?", "Not for the classifier: validation was already perfect at epoch ~12, so it would only "
         "memorise. Yes for YOLOv8s, which was still improving at epoch 80, but after adding production-camera data."),
        ("How will you reach high confidence?", "Calibrate scores (temperature scaling), use two limits per defect "
         "(confident defect / confident good) and reject the unsure middle; gate production on escape rate ≤ 0.1 %."),
        ("How accurate are your timestamps?", "Consistent but not exact: the Windows monotonic clock in Python 3.8 ticks "
         "every 15.6 ms, frames are stamped after read() returns, and the trigger after the serial reply. The plan "
         "switches to perf_counter, brackets each stamp and measures latency with an LED test."),
        ("What is the role of the model registry?", "No trained model reaches production until it passes a held-out test, "
         "a real-camera check and a shortcut check, and is approved; every activation is logged and reversible."),
        ("How do you add a new product?", "Images + labels, a detector trained on its parts, and a recipe in config.json; "
         "no code changes."),
    ]
    for q, a in qa:
        par = D.p(f"**Q: {q}**")
        par.paragraph_format.space_after = Pt(1)
        D.p(f"A: {a}", align="j")
    D.h("Part E - Checklist before the review", 1)
    D.bullets(["Fill names, roll numbers, guide and college on the title pages.",
               "Add application screenshots (Production, Label, Models, Database) to the black book.",
               "Set fault_action to REJECT before any live demo with the PLC.",
               "Print the results summary table and the upgrade-plan table as handouts.",
               "Rehearse the real-trial slides: present them as findings, not failures."])
    D.save("ABVIS_Final_Review_and_Presentation.docx")


if __name__ == "__main__":
    blackbook()
    review()
