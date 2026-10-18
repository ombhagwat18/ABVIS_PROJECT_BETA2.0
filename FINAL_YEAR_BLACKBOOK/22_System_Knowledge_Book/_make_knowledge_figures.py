"""Draws the explanatory figures of the System Knowledge Book (read-only for the project).

    python FINAL_YEAR_BLACKBOOK/22_System_Knowledge_Book/_make_knowledge_figures.py

k01_machine_layout       top view of the conveyor, photo-eye, cameras, reject cylinder, PLC and PC
k02_project_lifecycle    the whole loop: capture -> label -> split -> train -> gates -> line -> record -> back
k03_preprocessing        a real training image through dataset.model_input (ROI -> crop -> letterbox -> 192x448)
k04_training_pipeline    how a classifier checkpoint is trained and selected
k05_decision_logic       per-frame findings -> three-zone rule -> frame vote -> camera fusion -> one verdict
k06_plc_timing           signal timing of one PASS and one REJECT bottle with the saved T0 / T1
k07_machine_states       the machine states and what moves between them
k08_recipe_zones         how the detection recipe turns boxes into findings
k09_threads              the threads of the running application and how they talk
k10_annotation_review    Annotation Studio: proposal -> human decision -> export
k11_dataset_pipeline     folders -> labels.csv -> scenes -> split -> crop cache
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# drawing helpers and palette shared with the production-plan flowcharts (folder 21)
_spec = importlib.util.spec_from_file_location(
    "_plan_flow", ROOT / "FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_make_flowcharts.py")
PF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PF)
N, link = PF.N, PF.link
INK, DIM, BLUE, GREEN, RED, AMBER, GREY = PF.INK, PF.DIM, PF.BLUE, PF.GREEN, PF.RED, PF.AMBER, PF.GREY
F_DONE, F_SOFT, F_HW, F_FUT, F_GATE, F_BAD, F_TERM = PF.F_DONE, PF.F_SOFT, PF.F_HW, PF.F_FUT, PF.F_GATE, PF.F_BAD, PF.F_TERM

SETTINGS = json.loads((ROOT / "settings.json").read_text())


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ k01 machine layout
def k01_layout():
    speed = float(SETTINGS.get("conveyor_mm_s") or 0)
    dist = float(SETTINGS.get("inspection_to_reject_mm") or 0)
    t0, t1 = float(SETTINGS.get("plc_t0_s")), float(SETTINGS.get("plc_t1_s"))
    fig, ax = canvas(13.6, 7.2, 1950, 1000)
    # belt
    ax.add_patch(Rectangle((100, 420), 1524, 160, fc="#d5dbe1", ec=INK, lw=1.2))
    ax.text(110, 595, "Conveyor belt about 1524 mm (5 ft), no encoder", fontsize=9, color=DIM)
    for x in range(160, 1600, 120):
        ax.annotate("", xy=(x + 50, 500), xytext=(x, 500), arrowprops=dict(arrowstyle="-|>", color="#9aa5b1", lw=1))
    eye, rej = 330, 330 + dist
    # bottles (top view)
    for x, c in ((180, GREEN), (eye, BLUE), (760, GREEN), (1120, RED)):
        ax.add_patch(Circle((x, 500), 32, fc="white", ec=c, lw=2))
        ax.add_patch(Circle((x, 500), 12, fc=c, ec=c))
    # photo-eye
    ax.plot([eye, eye], [400, 600], color=AMBER, lw=1.5, ls="--")
    ax.add_patch(Rectangle((eye - 28, 360), 56, 40, fc=F_GATE, ec=INK))
    ax.text(eye, 330, "Photo-eye\nPLC input X0", ha="center", va="top", fontsize=9, fontweight="bold")
    # cameras
    ax.add_patch(Rectangle((eye - 45, 640), 90, 55, fc=F_SOFT, ec=INK))
    ax.text(eye - 60, 668, "Camera 1\n(EMEET Nova 4K)\nat the photo-eye", ha="right", va="center", fontsize=9,
            fontweight="bold")
    ax.annotate("", xy=(eye, 540), xytext=(eye, 640), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    c2 = eye + 220
    ax.add_patch(Rectangle((c2 - 45, 305), 90, 55, fc=F_SOFT, ec=INK, ls="--"))
    ax.text(c2 + 60, 250, "Camera 2 (planned): opposite wall,\nshifted toward the exit", ha="center", fontsize=8.5, color=DIM)
    ax.annotate("", xy=(c2, 460), xytext=(c2, 360), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2, ls="--"))
    ax.add_patch(Rectangle((eye - 120, 620), 360, 120, fc="none", ec=GREY, ls=":"))
    ax.text(eye - 115, 750, "Black matte enclosure + 2 LED strips 6500 K", fontsize=8, color=DIM)
    # reject cylinder
    ax.add_patch(Rectangle((rej - 40, 610), 80, 150, fc=F_BAD, ec=INK))
    ax.add_patch(Rectangle((rej - 12, 580), 24, 30, fc=RED, ec=INK))
    ax.text(rej, 790, "Reject cylinder\nFesto DSNU + 5/2 valve\nPLC output Y0", ha="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(rej, 330), xytext=(rej, 420), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(rej, 300, "rejected bottle\npushed off", ha="center", va="top", fontsize=8.5, color=RED)
    # distance
    ax.annotate("", xy=(rej, 880), xytext=(eye, 880), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.2))
    ax.text((eye + rej) / 2, 900, f"inspection -> reject  {dist:.0f} mm  /  belt {speed:.1f} mm/s  =  travel "
            f"{dist / speed:.1f} s" if speed else "travel not measured", ha="center", fontsize=10, fontweight="bold")
    # PLC + PC
    N(ax, "box", 1830, 520, 200, 150, "Delta DVP-SS2\nPLC ladder\nT0 = %.1f s  T1 = %.1f s\nowns Y0 / Y1" % (t0, t1),
      F_BAD, fs=9, bold=True)
    N(ax, "box", 1830, 180, 200, 170, "Windows PC\nNexaInspect app\nRTX 3050 4 GB\nPython 3.8", F_SOFT, fs=9, bold=True)
    ax.annotate("", xy=(1830, 445), xytext=(1830, 265), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3))
    ax.text(1650, 280, "RS-232\nModbus ASCII\n9600 7E1, COM5\nPC writes\nM0 / M1 only", fontsize=8, color=DIM)
    ax.plot([eye + 28, 1730], [380, 380], color=AMBER, lw=0.8, ls=":")
    ax.plot([rej + 40, 1730], [690, 690], color=RED, lw=0.8, ls=":")
    ax.plot([eye + 45, 1630], [660, 660], color=BLUE, lw=0.8, ls=":")
    ax.plot([1630, 1630, 1730], [660, 180, 180], color=BLUE, lw=0.8, ls=":")
    ax.text(1450, 130, "USB cameras -> PC", fontsize=8, color=BLUE)
    ax.text(100, 960, "Machine layout (top view, not to scale except the conveyor axis)", fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(fc=GREEN, label="good bottle"), Patch(fc=BLUE, label="bottle being inspected"),
                       Patch(fc=RED, label="bottle to reject")], loc="lower left", fontsize=8.5, frameon=False)
    save(fig, "k01_machine_layout.png")


# ------------------------------------------------------------------ k02 project lifecycle
def k02_lifecycle():
    fig, ax = canvas(12, 8.2, 12, 8.2)
    ax.text(0.2, 7.9, "The whole project as one loop: from a picture to a decision and back", fontsize=12, fontweight="bold")
    s = [  # x, y, text, fill
        (2.0, 6.6, "1  CAPTURE\nbottle images (good + each defect)\nvideo frames, Auto-collect on the line", F_HW),
        (6.0, 6.6, "2  LABEL\nLabel page: labels.csv (multi-label)\nAnnotate: boxes / polygons", F_SOFT),
        (10.0, 6.6, "3  SPLIT\nscene detection (near-duplicate frames)\nwhole scenes -> train / val / test", F_SOFT),
        (10.0, 4.2, "4  TRAIN\nEfficientNet-B0 classifier\nYOLOv8 component detector", F_SOFT),
        (10.0, 1.8, "5  EVALUATE\nheld-out test (read once)\nreal-camera check, shortcut check", F_GATE),
        (6.0, 1.8, "6  REGISTRY GATES\nCANDIDATE > VALIDATED >\nAPPROVED > ACTIVE (rollback)", F_GATE),
        (2.0, 1.8, "7  RUN THE LINE\ntrigger > frames > AI > decision\n> PLC M0 / M1 > cylinder", F_HW),
        (2.0, 4.2, "8  RECORD + LEARN\nSQLite, evidence, alarms, logs\nhard examples, new captures", F_DONE),
    ]
    nodes = [N(ax, "box", x, y, 3.4, 1.3, t, f, fs=8.6) for x, y, t, f in s]
    link(ax, nodes[0].r, nodes[1].l); link(ax, nodes[1].r, nodes[2].l); link(ax, nodes[2].b, nodes[3].t)
    link(ax, nodes[3].b, nodes[4].t); link(ax, nodes[4].l, nodes[5].r); link(ax, nodes[5].l, nodes[6].r)
    link(ax, nodes[6].t, nodes[7].b); link(ax, nodes[7].t, nodes[0].b, label="next dataset version", dx=0.1, dy=0.5, col=BLUE)
    N(ax, "box", 6.0, 4.2, 3.6, 1.5, "RULES THAT HOLD THE LOOP TOGETHER\nunknown is FAULT, never PASS\n"
      "a proposal is not a label until accepted\ntest set never chooses anything\ntraining never activates a model",
      "white", fs=8.3, ec=RED, ls="--")
    ax.legend(handles=[Patch(fc=F_HW, ec=INK, label="needs camera / machine"), Patch(fc=F_SOFT, ec=INK, label="software"),
                       Patch(fc=F_GATE, ec=INK, label="quality gate"), Patch(fc=F_DONE, ec=INK, label="traceability")],
              loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, "k02_project_lifecycle.png")


# ------------------------------------------------------------------ k03 preprocessing (real image)
def k03_preprocessing():
    sys.path.insert(0, str(ROOT))
    from vision import dataset as D          # the one crop path used by training AND inference
    # the ROI baked into the active checkpoint 20260919-164511 (model.pt), read once with torch
    cfg = {"roi": [826, 0, 841, 1927], "roi_frame": [2537, 1927], "input_wh": [192, 448], "cache_wh": [216, 496]}
    src = sorted((ROOT / "projects/om_bottle/images/-ve/Skewed Label").glob("*.jpg"))[5]
    full = cv2.imread(str(src))
    x, y, w, h = cfg["roi"]
    boxed = full.copy(); cv2.rectangle(boxed, (x, y), (x + w, y + h), (40, 40, 220), 14)
    cropped = D.crop(full, cfg["roi"], cfg["roi_frame"])
    letter = D.letterbox(cropped, cfg["cache_wh"])
    final = D.model_input(full, cfg)
    panels = [(boxed, f"1  Full camera frame\n{full.shape[1]} x {full.shape[0]} px, ROI in red"),
              (cropped, f"2  ROI crop\n{cropped.shape[1]} x {cropped.shape[0]} px"),
              (letter, f"3  Letterbox to cache size\n{letter.shape[1]} x {letter.shape[0]} (PNG cache)"),
              (final, f"4  Centre crop = model input\n{final.shape[1]} x {final.shape[0]} (tall, not square)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 5.6), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    for a, (im, t) in zip(axs, panels):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.set_title(t, fontsize=9.5); a.axis("off")
    fig.suptitle("Image pre-processing: one function (dataset.model_input) for training and live inference  "
                 f"[example: {src.parent.name}/{src.name}]", fontsize=11, fontweight="bold")
    save(fig, "k03_preprocessing.png")


# ------------------------------------------------------------------ k04 training pipeline
def k04_training():
    fig, ax = canvas(10.5, 13.5, 10.5, 13.5)
    ax.text(0.2, 13.2, "How a classifier checkpoint is trained and chosen (vision/train.py)", fontsize=12, fontweight="bold")
    X = 5.25
    steps = [
        ("term", "labels.csv  (reviewed = 1 rows only)\nunreviewed images never train", F_TERM),
        ("box", "Scene map: consecutive near-duplicate frames grouped\nsplit by whole scenes ~70 / 15 / 15  (train / val / test)", F_SOFT),
        ("box", "Pre-process: ROI crop > letterbox > centre crop 192 x 448\nPNG crop cache (JPEG moved water_level by ~0.15)", F_SOFT),
        ("box", "Augment (train only): small rotation / scale, brightness / contrast,\nperspective, blur, colour, noise.  NEVER horizontal flip", F_SOFT),
        ("box", "Sampler: good bottles up to x5, hard examples x3\n(WeightedRandomSampler; recorded in metrics.json)", F_SOFT),
        ("box", "EfficientNet-B0 (ImageNet weights) + 1 linear head, 8 sigmoid outputs\n"
                "BCEWithLogits, pos_weight = clip(neg / pos, 1, 50)\nAdamW lr 3e-4, cosine schedule, batch 32, 25 epochs, AMP", F_DONE),
        ("box", "Each epoch: validation loss + macro-F1\nkeep best macro-F1 epoch, tie > lowest val loss", F_GATE),
        ("box", "Per defect: threshold = middle of the best-F1 band on val\n"
                "zero training positives > threshold 1.01 (disabled)\ntemperature fitted on val logits (calibration)", F_GATE),
        ("box", "Save models/<stamp>/: model.pt (weights + ROI + defects + input size)\n"
                "metrics.json (history, val_paths, test_paths), labels.csv copy", F_DONE),
        ("term", "CANDIDATE in the registry: config.json NOT changed\ntest set scored once (registry.model_bench cls-test)", F_TERM),
    ]
    ys = [12.4 - i * 1.3 for i in range(len(steps))]
    nodes = [N(ax, k, X, y, 8.6, 1.0 if "\n" in t and t.count("\n") > 1 else 0.85, t, f, fs=8.6) for (k, t, f), y in zip(steps, ys)]
    for a, b in zip(nodes, nodes[1:]):
        link(ax, a.b, b.t)
    save(fig, "k04_training_pipeline.png")


# ------------------------------------------------------------------ k05 decision logic
def k05_decision():
    fig, ax = canvas(12.2, 12.5, 12.2, 12.5)
    ax.text(0.2, 12.2, "Decision engine (line/decision.py): many frames, many cameras > one verdict", fontsize=12,
            fontweight="bold")
    a = N(ax, "term", 6, 11.3, 6.2, 0.8, "Frames of ONE bottle from ONE camera\n(stamped strictly after the trigger, inside its window)", F_TERM, fs=8.8)
    b = N(ax, "box", 3.0, 9.6, 5.2, 1.4, "Classifier per frame\nscore of each defect (after temperature)\n"
          ">= threshold : DEFECT\n<= pass_max 0.10 : absent\nin between : UNSURE", F_SOFT, fs=8.5)
    c = N(ax, "box", 9.0, 9.6, 5.2, 1.4, "Detector + recipe per frame\npick the bottle (upright, >= 2 % area,\nnearest station)"
          " ; required part not in its zone\n> missing_cap / missing_label ; cap_tilted box\n= defect (recipe 'defects')", F_SOFT, fs=8.5)
    d = N(ax, "dia", 6, 7.6, 5.4, 1.5, "Bottle seen in >= 60 %\nof frames, models OK?", F_GATE, fs=8.8)
    e = N(ax, "box", 10.2, 7.6, 2.8, 1.1, "FAULT\n(no bottle / stale /\nmodel error)", F_BAD, fs=8.5, bold=True)
    f = N(ax, "box", 6, 5.8, 6.4, 1.1, "Frame vote per defect (majority)\na defect counts only if it is in most of the bottle's frames", F_SOFT, fs=8.6)
    g = N(ax, "dia", 6, 4.0, 4.6, 1.5, "Camera result?", F_GATE, fs=9)
    h1 = N(ax, "box", 1.6, 2.4, 2.6, 0.9, "PASS\nevery defect absent", F_DONE, fs=8.5, bold=True)
    h2 = N(ax, "box", 6, 2.4, 2.6, 0.9, "REJECT\na voted defect", F_BAD, fs=8.5, bold=True)
    h3 = N(ax, "box", 10.4, 2.4, 2.6, 0.9, "FAULT (UNSURE)\ncounted + alarm", F_BAD, fs=8.5, bold=True)
    z = N(ax, "box", 6, 0.75, 9.0, 0.9, "Fuse the cameras:  FAULT > REJECT > PASS\nPASS only if EVERY camera saw the bottle and passed it",
          F_TERM, fs=9, bold=True)
    link(ax, a.b, (6, 10.6), (3.0, 10.6), b.t); link(ax, (6, 10.6), (9.0, 10.6), c.t)
    link(ax, b.b, (3.0, 8.6), (6, 8.6), d.t); link(ax, c.b, (9.0, 8.6), (6, 8.6))
    link(ax, d.r, e.l, label="No", dx=0.15, dy=0.25); link(ax, d.b, f.t, label="Yes", dx=0.1, dy=-0.3)
    link(ax, f.b, g.t)
    link(ax, g.l, (1.6, 4.0), h1.t, label="all absent", dx=-1.6, dy=0.1)
    link(ax, g.b, h2.t, label="defect", dx=0.1, dy=-0.35)
    link(ax, g.r, (10.4, 4.0), h3.t, label="only unsure", dx=0.15, dy=0.1)
    for n in (h1, h2, h3):
        link(ax, n.b, (n.x, 1.2))
    link(ax, e.r, (11.95, e.y), (11.95, 0.75), z.r)
    save(fig, "k05_decision_logic.png")


# ------------------------------------------------------------------ k06 PLC timing diagram
def k06_timing():
    t0, t1 = float(SETTINGS.get("plc_t0_s")), float(SETTINGS.get("plc_t1_s"))
    speed, dist = float(SETTINGS.get("conveyor_mm_s") or 0), float(SETTINGS.get("inspection_to_reject_mm") or 0)
    travel = dist / speed if speed else 9.5
    send = travel - t0
    end = travel + t1
    fig, ax = plt.subplots(figsize=(13, 7.4))
    sig = ["X0 photo-eye", "M2 trigger (PLC)", "M0 PASS (PC)", "M1 REJECT (PC)", "T0 timer", "Y0 cylinder"]
    y = {s: (len(sig) - i) * 1.4 for i, s in enumerate(sig)}

    def wave(name, pts, col=INK):
        b = y[name]
        xs, ys = [-1], [b]
        for a, c in pts:
            xs += [a, a, c, c]; ys += [b, b + 0.8, b + 0.8, b]
        xs.append(end + 3); ys.append(b)
        ax.plot(xs, ys, color=col, lw=2, drawstyle="default")

    # bottle A = PASS at 0 s (answered at 0.6 s), bottle B = REJECT at 2 s, bottle C arrives during the reject cycle
    tb = 2.0
    wave("X0 photo-eye", [(0, 0.3), (tb, tb + 0.3), (tb + 6.5, tb + 6.8)], AMBER)
    wave("M2 trigger (PLC)", [(0.02, 0.6), (tb + 0.02, tb + send)], BLUE)
    wave("M0 PASS (PC)", [(0.6, 0.65)], GREEN)
    wave("M1 REJECT (PC)", [(tb + send, tb + end)], RED)
    wave("T0 timer", [(tb + send, tb + travel)], GREY)
    wave("Y0 cylinder", [(tb + travel, tb + end)], RED)
    for s in sig:
        ax.text(-1.2, y[s] + 0.35, s, ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.axvline(tb + 6.5, color=RED, ls=":", lw=1)
    ax.text(tb + 6.6, y["X0 photo-eye"] + 1.0, "bottle C: no trigger while M1 is ON\n> recorded NOT INSPECTED (FAULT)",
            fontsize=8.5, color=RED)
    ax.text(0.8, y["M0 PASS (PC)"] + 0.15, "bottle A PASS:\nM0 written at once,\nPLC clears M0 + M2", fontsize=8.5, color=GREEN)
    ax.annotate("", xy=(tb + send, 0.4), xytext=(tb, 0.4), arrowprops=dict(arrowstyle="<|-|>", color=INK))
    ax.text(tb + send / 2, 0.55, f"travel - T0 = {send:.1f} s", ha="center", fontsize=9)
    ax.annotate("", xy=(tb + travel, -0.2), xytext=(tb, -0.2), arrowprops=dict(arrowstyle="<|-|>", color=INK))
    ax.text(tb + travel / 2, -0.05, f"travel = {dist:.0f} mm / {speed:.1f} mm/s = {travel:.1f} s (bottle B reaches the cylinder)",
            ha="center", fontsize=9)
    ax.text(tb + travel + t1 / 2, y["Y0 cylinder"] + 1.0, f"T1 = {t1:.1f} s", ha="center", fontsize=9, color=RED)
    ax.text(tb + send + t0 / 2, y["T0 timer"] + 1.0, f"T0 = {t0:.1f} s", ha="center", fontsize=9)
    ax.set_xlim(-6.5, end + tb + 1.5); ax.set_ylim(-0.6, len(sig) * 1.4 + 2.2)
    ax.set_xticks(range(0, int(end + tb) + 2, 2)); ax.set_yticks([]); ax.set_xlabel("time after bottle A's trigger (s)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("PLC handshake timing with the saved settings (T0 = %.1f s, T1 = %.1f s); bottle B is a REJECT" % (t0, t1),
                 fontsize=11.5, fontweight="bold", loc="left")
    save(fig, "k06_plc_timing.png")


# ------------------------------------------------------------------ k07 machine states
def k07_states():
    fig, ax = canvas(12, 7.8, 12, 7.8)
    ax.text(0.2, 7.5, "Machine states (line/machine_state.py): one state, shown on every screen", fontsize=12, fontweight="bold")
    off = N(ax, "box", 1.4, 6.2, 2.2, 0.8, "OFFLINE\nPLC not connected", F_FUT, fs=8.5, bold=True)
    nr = N(ax, "box", 4.6, 6.2, 2.4, 0.8, "NOT_READY\na start check fails", F_GATE, fs=8.5, bold=True)
    rd = N(ax, "box", 8.0, 6.2, 2.4, 0.8, "READY\nSTART enabled", F_DONE, fs=8.5, bold=True)
    ini = N(ax, "box", 10.6, 4.4, 2.2, 0.9, "INITIALIZING\nmodels, cameras,\nwarm-up", F_SOFT, fs=8.5, bold=True)
    run = N(ax, "box", 7.6, 3.0, 2.4, 0.8, "RUNNING\nwaiting for a bottle", F_DONE, fs=8.5, bold=True)
    ins = N(ax, "box", 4.0, 3.0, 2.6, 0.8, "INSPECTING\nbottle(s) in progress", F_DONE, fs=8.5, bold=True)
    stp = N(ax, "box", 10.6, 1.6, 2.2, 0.8, "STOPPING\nfinish bottles", F_SOFT, fs=8.5, bold=True)
    flt = N(ax, "box", 1.5, 1.0, 2.4, 0.9, "FAULT\nhalt latch / error\nRESET FAULT", F_BAD, fs=8.5, bold=True)
    es = N(ax, "box", 4.5, 1.0, 2.4, 0.9, "E_STOP\nE-stop input\npressed", F_BAD, fs=8.5, bold=True)
    cf = N(ax, "box", 7.6, 1.0, 2.6, 0.9, "COMMUNICATION_FAULT\nPLC link lost while\nrunning", F_BAD, fs=8.3, bold=True)
    link(ax, off.r, nr.l, label="connect", dx=0.05, dy=0.12)
    link(ax, nr.r, rd.l, label="all checks OK", dx=0.02, dy=0.12)
    link(ax, rd.r, (10.6, 6.2), ini.t, label="START", dx=0.1, dy=0.1)
    link(ax, ini.b, (10.6, 3.0), run.r, label="every camera fresh", dx=-1.5, dy=-0.35)
    link(ax, run.l, ins.r, label="trigger", dx=-0.95, dy=0.12)
    link(ax, (ins.x, ins.y - 0.4), (ins.x, 2.2), (run.x - 0.3, 2.2), (run.x - 0.3, run.y - 0.4))
    ax.text(5.0, 2.3, "verdict sent", fontsize=8, color=INK, fontweight="bold")
    link(ax, (run.x + 1.2, 2.8), (stp.x, 2.8), stp.t, label="STOP", dx=0.1, dy=-0.3, lpos=1)
    ax.text(0.3, 1.75, "Any state > these three when their condition is true (checked first):", fontsize=8.3,
            color=RED, fontweight="bold")
    link(ax, stp.b, (10.6, 0.3), (8.0 + 3.0, 0.3), col=GREY)
    ax.text(9.2, 0.38, "> READY / NOT_READY", fontsize=8, color=DIM)
    ax.text(0.3, 0.1, "START checklist: PLC connected and in RUN, cameras chosen, models present, timing valid "
            "(T0 < travel, speed measured), FAULT -> REJECT, E-stop released, recipe valid.", fontsize=8.3, color=DIM)
    save(fig, "k07_machine_states.png")


# ------------------------------------------------------------------ k08 recipe zones
def k08_recipe():
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 6.2))
    cases = [("Complete bottle > PASS", True, True, GREEN), ("No cap box in the cap zone > missing_cap", False, True, RED),
             ("No label box in the label zone > missing_label", True, False, RED)]
    for ax, (title, cap, label, col) in zip(axs, cases):
        ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off"); ax.set_aspect("equal")
        ax.add_patch(FancyBboxPatch((3, 1), 4, 14, boxstyle="round,pad=0.1,rounding_size=1", fc="#e8f1f8", ec=GREY))
        ax.add_patch(Rectangle((4.1, 15), 1.8, 1.4, fc="#e8f1f8", ec=GREY))
        if cap:
            ax.add_patch(Rectangle((3.9, 16.4), 2.2, 1.4, fc="#4a90d9", ec=INK))
        ax.add_patch(Rectangle((2.6, 0.6), 4.8, 17.8 if cap else 16.0, fc="none", ec=GREEN, lw=2))
        ax.text(2.6, 18.9 if cap else 17.0, "bottle (anchor)", color=GREEN, fontsize=8.5)
        ax.add_patch(Rectangle((2.2, 9.5), 5.6, 9.8, fc=AMBER, alpha=0.08, ec=AMBER, ls="--"))
        ax.text(2.1, 14.5, "cap zone:\ntop 50 % of\nthe bottle\n(+25 %\nabove)", fontsize=7.5, color=AMBER, ha="right")
        if cap:
            ax.add_patch(Rectangle((3.8, 16.3), 2.4, 1.6, fc="none", ec=BLUE, lw=2)); ax.text(6.4, 16.8, "cap", color=BLUE, fontsize=8.5)
        if label:
            ax.add_patch(Rectangle((3, 5.5), 4, 4.5, fc="#f3c26b", ec=INK))
            ax.add_patch(Rectangle((2.9, 5.4), 4.2, 4.7, fc="none", ec="#8e44ad", lw=2))
            ax.text(7.6, 7.5, "label", color="#8e44ad", fontsize=8.5, zorder=5)
        ax.set_title(title, fontsize=9.5, color=col, fontweight="bold")
    fig.suptitle("Detection recipe (config.json 'inspection'): anchor + required parts + zones, judged per frame then voted",
                 fontsize=11, fontweight="bold")
    save(fig, "k08_recipe_zones.png")


# ------------------------------------------------------------------ k09 threads
def k09_threads():
    fig, ax = canvas(12, 7.6, 12, 7.6)
    ax.text(0.2, 7.3, "Threads inside the running application and how they talk", fontsize=12, fontweight="bold")
    tk = N(ax, "box", 6, 6.2, 5.0, 1.0, "Tk main thread (the only one that touches widgets)\nApp.pump() drains App.q every 60 ms, runs GC",
           F_SOFT, fs=8.6, bold=True)
    g1 = N(ax, "box", 1.8, 4.2, 3.0, 1.1, "Camera grab thread x N\nclock.now() stamp + seq\nkeeps the newest frames", F_DONE, fs=8.4)
    sc = N(ax, "box", 1.8, 2.2, 3.0, 1.0, "Live score thread\n(view only, ~15 Hz,\nnewest frame, skips old)", F_DONE, fs=8.4)
    mc = N(ax, "box", 6, 4.2, 3.4, 1.1, "Machine-cycle thread\ntriggers, frame windows, FIFO,\ndeadlines (never runs a model)", F_GATE, fs=8.4, bold=True)
    ai = N(ax, "box", 6, 2.2, 3.4, 1.0, "AI worker (async_ai)\nclassifier + YOLO per frame", F_GATE, fs=8.4)
    plc = N(ax, "box", 10.2, 4.2, 3.0, 1.1, "PLCService thread\nonly owner of the link\nfast poll M2 / X0 / Y0", F_BAD, fs=8.4, bold=True)
    ev = N(ax, "box", 10.2, 2.2, 3.0, 1.0, "Evidence writer\nimages to disk, bounded,\nnever delays the PLC", F_FUT, fs=8.4)
    db = N(ax, "box", 6, 0.6, 5.2, 0.8, "SQLite production.db  +  logs/<channel>.log  +  daily CSV", F_TERM, fs=8.6)
    link(ax, g1.r, mc.l); ax.text(3.45, 4.3, "frames", fontsize=8.5, fontweight="bold")
    link(ax, g1.b, sc.t)
    link(ax, (mc.x - 0.5, mc.y - 0.55), (ai.x - 0.5, ai.y + 0.5))
    link(ax, (ai.x + 0.5, ai.y + 0.5), (mc.x + 0.5, mc.y - 0.55), col=GREY)
    ax.text(4.0, 3.2, "frames to score", fontsize=8); ax.text(6.6, 3.2, "scores back", fontsize=8, color=DIM)
    link(ax, (plc.x - 1.5, 4.4), (mc.x + 1.7, 4.4)); ax.text(7.8, 4.5, "Trigger", fontsize=8.5, fontweight="bold")
    link(ax, (mc.x + 1.7, 3.9), (plc.x - 1.5, 3.9), col=RED)
    ax.text(7.75, 3.6, "M0 / M1 command", fontsize=8, color=RED, fontweight="bold")
    link(ax, mc.t, tk.b, col=BLUE); ax.text(6.1, 5.15, "App.post(...)", fontsize=8.5, color=BLUE, fontweight="bold")
    link(ax, ev.b, (10.2, 0.6), db.r); link(ax, (6, 1.7), db.t)
    link(ax, (mc.x + 1.7, mc.y - 0.45), (8.4, mc.y - 0.45), (8.4, 2.2), ev.l)
    ax.text(8.5, 2.9, "evidence", fontsize=8, color=DIM)
    save(fig, "k09_threads.png")


# ------------------------------------------------------------------ k10 annotation review
def k10_annotation():
    fig, ax = canvas(12, 6.6, 12, 6.6)
    ax.text(0.2, 6.3, "Annotation Studio: a model may propose, only a person decides", fontsize=12, fontweight="bold")
    a = N(ax, "box", 1.7, 4.6, 2.8, 1.1, "Unreviewed images\n(projects/<slug>/\nannotations.json)", F_TERM, fs=8.5)
    b = N(ax, "box", 5.2, 4.6, 3.0, 1.1, "Background thread:\ndetector proposes boxes\n(model id + confidence)", F_SOFT, fs=8.5)
    c = N(ax, "dia", 9.2, 4.6, 3.6, 1.6, "Person reviews\neach proposal", F_GATE, fs=9)
    d1 = N(ax, "box", 7.0, 2.2, 2.2, 0.9, "ACCEPT\n> becomes a box", F_DONE, fs=8.5, bold=True)
    d2 = N(ax, "box", 9.4, 2.2, 2.2, 0.9, "CORRECT\n(original kept)", F_DONE, fs=8.5, bold=True)
    d3 = N(ax, "box", 11.0, 3.5, 1.7, 0.8, "REJECT\nhidden, undoable", F_BAD, fs=8, bold=True)
    e = N(ax, "box", 4.0, 2.2, 3.2, 0.9, "Atomic save +\nannotation_log.jsonl (audit)", F_FUT, fs=8.5)
    f = N(ax, "box", 4.0, 0.6, 5.6, 0.8, "Export YOLO labels: only reviewed images or accepted boxes;\nan image still holding proposals is skipped",
          F_TERM, fs=8.4, bold=True)
    link(ax, a.r, b.l); link(ax, b.r, c.l)
    link(ax, (c.x - 0.8, c.y - 0.4), d1.t); link(ax, c.b, (9.4, 2.65)); link(ax, (c.x + 1.2, c.y - 0.2), d3.t)
    link(ax, d1.l, e.r); link(ax, (d2.x, 1.75), (d2.x, 1.4), (6.8, 1.4), (6.8, 1.0))
    link(ax, e.b, f.t)
    ax.text(0.3, 3.0, "Queues reorder unreviewed images:\nleast sure / missing part /\ndoubtful part (hard negatives)",
            fontsize=8.3, color=DIM)
    save(fig, "k10_annotation_review.png")


# ------------------------------------------------------------------ k11 dataset pipeline
def k11_dataset():
    fig, ax = canvas(12, 5.6, 12, 5.6)
    ax.text(0.2, 5.3, "Classification data: folders seed labels once, labels.csv is the truth", fontsize=12, fontweight="bold")
    a = N(ax, "box", 1.7, 3.6, 3.0, 1.6, "images/\n+ve/  > good, reviewed\n-ve/<Class>/ > that defect\n-ve/ or _inbox/ > unreviewed",
          F_TERM, fs=8.4)
    b = N(ax, "box", 5.3, 3.6, 3.2, 1.6, "labels.csv\npath, 8 defect columns, reviewed\nmulti-label; GOOD = all 0\nedits logged + undoable",
          F_DONE, fs=8.4, bold=True)
    c = N(ax, "box", 8.9, 3.6, 3.0, 1.6, "scenes.json\nnear-duplicate video frames\n= one physical bottle pass", F_SOFT, fs=8.4)
    d = N(ax, "box", 8.9, 1.0, 3.0, 1.2, "train / val / test\nby whole scene\ntest read once", F_GATE, fs=8.4, bold=True)
    e = N(ax, "box", 5.3, 1.0, 3.2, 1.2, "cache/: PNG crops 216 x 496\n> centre crop 192 x 448", F_SOFT, fs=8.4)
    g = N(ax, "box", 1.7, 1.0, 3.0, 1.2, "Model pre-labels\ncache/suggestions.json\nnever labels until accepted", F_FUT, fs=8.4)
    link(ax, a.r, b.l, label="first import only", dx=-1.0, dy=0.12)
    link(ax, b.r, c.l); link(ax, c.b, d.t); link(ax, d.l, e.r); link(ax, g.t, (1.7, 2.2), (4.6, 2.2), (4.6, 2.8))
    ax.text(1.85, 2.3, "person accepts", fontsize=8, color=BLUE, fontweight="bold")
    save(fig, "k11_dataset_pipeline.png")


if __name__ == "__main__":
    k01_layout(); k02_lifecycle(); k03_preprocessing(); k04_training(); k05_decision(); k06_timing()
    k07_states(); k08_recipe(); k09_threads(); k10_annotation(); k11_dataset()
