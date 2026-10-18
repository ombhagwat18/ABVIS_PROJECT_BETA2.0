"""Builds every figure of the final report from the files on disk (read-only for the project).

    python FINAL_YEAR_BLACKBOOK/20_Final_Report/_make_figures.py

Sources: projects/om_bottle/{labels.csv, models/*/metrics.json, test_metrics.json}, models_archive (active model),
models/stage2_yolo (results.csv, candidate_*.json), vision_dataset/manifests/unified_manifest.csv,
projects/om_bottle/production/production.db, data/real_camera_model_check.json (real EMEET frames, 2026-10-13).
"""
from __future__ import annotations

import collections
import csv
import json
import shutil
import sqlite3
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
DATA = HERE / "data"

INK, DIM, GRID = "#1f2933", "#616e7c", "#d9dee4"
BLUE, GREEN, RED, AMBER, PURPLE, TEAL, GREY = "#2f6fb0", "#2e8b57", "#c0392b", "#d68910", "#7d3c98", "#138d90", "#95a5a6"
ARCH_COL = {"efficientnet_b0": BLUE, "resnet18": AMBER, "efficientnet_b1": PURPLE,
            "mobilenet_v3_small": TEAL, "convnext_tiny": RED}
NICE = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet18", "efficientnet_b1": "EfficientNet-B1",
        "mobilenet_v3_small": "MobileNetV3-S", "convnext_tiny": "ConvNeXt-T"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": DIM, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 150, "savefig.bbox": "tight"})
CLASSES = ["damaged_bottle", "damaged_label", "missing_cap", "missing_label", "skewed_bottle", "skewed_label",
           "tilt_cap", "water_level"]


def save(fig, name):
    # the document numbers its figures; drop the "Figure N - " prefix used while drafting
    import re
    strip = lambda s: re.sub(r"^Figure \d+ - ", "", s)
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            t = ax.get_title(loc)
            if t.startswith("Figure"):
                ax.set_title(strip(t), loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text(strip(fig._suptitle.get_text()))
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)
    print("wrote", name)


def jload(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ data
MODELS = ROOT / "projects/om_bottle/models"
cls_runs = []
for d in sorted(MODELS.glob("*/metrics.json")):
    m, t = jload(d), jload(d.parent / "test_metrics.json")
    cls_runs.append({"stamp": m["stamp"], "arch": m["arch"], "m": m, "t": t})
ACTIVE = ROOT / "models_archive/previous/om_bottle_models/20260919-164511"
active = {"stamp": "20260919-164511", "arch": "efficientnet_b0", "m": jload(ACTIVE / "metrics.json"),
          "t": jload(ACTIVE / "test_metrics.json")}


def yolo_rows(run):
    rows = list(csv.DictReader(open(ROOT / f"models/stage2_yolo/{run}/results.csv")))
    return [{k.strip(): float(v) for k, v in r.items()} for r in rows]


# ------------------------------------------------------------------ fig01 architecture
def box(ax, x, y, w, h, text, fc, ec=None, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc,
                                ec=ec or INK, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text="", col=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=7.5, color=DIM)


def fig01():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 2.25), 8.6, 5.0, boxstyle="round,pad=0.02", fc="#f4f7fb", ec=BLUE, lw=1.4, ls="--"))
    ax.text(0.4, 7.0, "Windows PC  -  Python 3.8 desktop application (ABVIS)", fontsize=10, color=BLUE, fontweight="bold")
    box(ax, 0.5, 5.6, 2.6, 1.0, "HMI / UI\nui/app.py, ui/hmi.py\n15 pages, operator / engineer", "#e3edf8")
    box(ax, 3.4, 5.6, 2.4, 1.0, "Camera layer\nvision/infer.py\ngrab + score threads", "#e8f6ef")
    box(ax, 6.1, 5.6, 2.5, 1.0, "AI models\nclassifier (EfficientNet)\ndetector (YOLOv8)", "#e8f6ef")
    box(ax, 0.5, 4.0, 2.6, 1.1, "Model registry\nregistry/\ncandidate > validated >\napproved > active", "#f6f0fa")
    box(ax, 3.4, 4.0, 2.4, 1.1, "Machine cycle\nline/machine_cycle.py\nInspector + FIFO +\ndeadlines", "#fdf2e3", bold=True)
    box(ax, 6.1, 4.0, 2.5, 1.1, "Decision engine\nline/decision.py\nrecipe, vote, fusion\nPASS / REJECT / FAULT", "#fdf2e3")
    box(ax, 0.5, 2.5, 2.6, 1.0, "Production record\nSQLite production.db\nevidence images", "#eef1f4")
    box(ax, 3.4, 2.5, 2.4, 1.0, "PLC service\nplc/service.py\nonly writes M0 / M1", "#fbe9e7", bold=True)
    box(ax, 6.1, 2.5, 2.5, 1.0, "Alarms, logs, state\nline/alarms.py, applog.py\nmachine_state.py", "#eef1f4")
    arrow(ax, 4.6, 5.6, 4.6, 5.1)
    ax.text(4.7, 5.3, "frames + timestamps", fontsize=7.5, color=DIM)
    arrow(ax, 5.8, 4.55, 6.1, 4.55)
    arrow(ax, 6.1, 4.3, 5.8, 4.3)
    arrow(ax, 7.35, 5.6, 7.35, 5.1)
    arrow(ax, 4.6, 4.0, 4.6, 3.5)
    arrow(ax, 3.4, 4.3, 3.1, 3.2)
    arrow(ax, 1.8, 4.0, 1.8, 3.5)
    arrow(ax, 1.8, 5.6, 1.8, 5.1)
    arrow(ax, 5.8, 3.0, 6.1, 3.0)
    # field devices
    box(ax, 3.3, 0.5, 2.6, 1.2, "Delta DVP-SS2 PLC\nladder: T0 delay, T1 pulse\nowns Y0 / Y1", "#fbe9e7", bold=True)
    arrow(ax, 4.6, 2.5, 4.6, 1.7)
    ax.text(4.7, 1.95, "Modbus ASCII, RS-232 9600 7E1", fontsize=7.5, color=DIM)
    box(ax, 0.3, 0.5, 2.4, 1.2, "Photo-eye X0\nStart X1 / Stop X2", "#ffffff")
    box(ax, 6.5, 0.5, 2.4, 1.2, "Reject cylinder Y0\n(Festo DSNU + 5/2 valve)\nConveyor Y1", "#ffffff")
    arrow(ax, 2.7, 1.1, 3.3, 1.1)
    arrow(ax, 5.9, 1.1, 6.5, 1.1)
    box(ax, 9.3, 5.6, 2.5, 1.0, "2 x EMEET Nova 4K\nUSB, 1920x1080 MJPG\nrotated 90 deg", "#ffffff")
    arrow(ax, 9.3, 6.1, 8.6, 6.1)
    box(ax, 9.3, 4.0, 2.5, 1.1, "Datasets\nom_bottle (1,202 img)\nstage2 (594 img,\n1,994 boxes)", "#ffffff")
    arrow(ax, 9.3, 4.55, 8.6, 4.85, "training")
    box(ax, 9.3, 2.5, 2.5, 1.0, "Conveyor belt\n108.2 mm/s, 1028 mm\ncamera > cylinder", "#ffffff")
    ax.set_title("Figure 1 - ABVIS system architecture", loc="left")
    save(fig, "fig01_architecture.png")


# ------------------------------------------------------------------ fig02 machine cycle
def fig02():
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    lanes = ["Photo-eye X0", "PLC ladder", "PC: PLC service", "PC: Inspector + AI", "PC: decision + FIFO", "Cylinder Y0"]
    xs = np.linspace(0.9, 11.1, len(lanes))
    for x, n in zip(xs, lanes):
        ax.add_patch(FancyBboxPatch((x - 0.85, 5.9), 1.7, 0.5, boxstyle="round,pad=0.02", fc="#e3edf8", ec=INK))
        ax.text(x, 6.15, n, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.plot([x, x], [0.3, 5.9], color=GRID, lw=1.5, zorder=0)
    steps = [(0, 1, 5.5, "bottle breaks beam"), (1, 2, 5.0, "SET M2 (inspect now)"),
             (2, 3, 4.5, "Trigger(id, t) - window opens"), (3, 4, 3.9, "frames after t > classifier + YOLO"),
             (4, 2, 3.3, "PASS: write M0 now"), (2, 1, 2.8, "M0 -> PLC clears M0, M2"),
             (4, 2, 2.2, "REJECT/FAULT: write M1 at t + travel - T0"), (2, 1, 1.7, "M1 -> T0 timer starts"),
             (1, 5, 1.1, "after T0: Y0 ON for T1 (cylinder)"), (5, 4, 0.55, "Y0 observed -> bottle recorded")]
    for a, b, y, t in steps:
        col = RED if "REJECT" in t or "Y0" in t else GREEN if "PASS" in t or "M0" in t else INK
        arrow(ax, xs[a], y, xs[b], y, col=col)
        ax.text((xs[a] + xs[b]) / 2, y + 0.08, t, ha="center", fontsize=8, color=col)
    ax.set_title("Figure 2 - Per-bottle machine cycle (trigger > inspection > decision > PLC command)", loc="left")
    save(fig, "fig02_machine_cycle.png")


# ------------------------------------------------------------------ fig03 throughput budget
def fig03():
    travel, t0, t1, win, ai = 1028 / 108.2, 5.0, 2.5, 3.9, 0.4
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.barh(2, win, color=BLUE, label="watch window (3.9 s)")
    ax.barh(2, ai, left=win, color=PURPLE, label="AI + vote (~0.4 s)")
    ax.text(win + ai + 0.1, 2, "M0 sent: M2 busy ~4.3 s", va="center", fontsize=9)
    ax.barh(1, travel - t0, color=GREY, label="wait until travel - T0")
    ax.barh(1, t0, left=travel - t0, color=AMBER, label="T0 delay (5.0 s)")
    ax.barh(1, t1, left=travel, color=RED, label="T1 pulse (2.5 s)")
    ax.text(travel + t1 + 0.1, 1, "M2 blocked ~12 s", va="center", fontsize=9)
    for k in range(1, 13):
        ax.axvline(k * 0.92, ymin=0, ymax=0.18, color=GREEN, lw=2)
    ax.text(11.4, 0.22, "green ticks = bottle arrivals,\n100 mm pitch (~0.92 s)", color=GREEN, fontsize=8.5,
            va="center")
    ax.set_yticks([1, 2]); ax.set_yticklabels(["REJECT bottle", "PASS bottle"])
    ax.set_xlim(0, 15); ax.set_ylim(0, 2.6); ax.set_xlabel("seconds after the trigger")
    ax.legend(ncol=3, fontsize=8, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Figure 3 - Why bottles went uninspected: one-bottle handshake time budget", loc="left")
    save(fig, "fig03_throughput_budget.png")


# ------------------------------------------------------------------ fig04 dataset
def labels_counts():
    rows = list(csv.reader(open(ROOT / "projects/om_bottle/labels.csv")))
    h, body = rows[0], [r for r in rows[1:] if r[-1] == "1"]
    cnt = {c: sum(1 for r in body if r[i] == "1") for i, c in enumerate(h[1:-1], 1)}
    cnt["good"] = sum(1 for r in body if all(x == "0" for x in r[1:-1]))
    return cnt, len(rows) - 1, len(body)


def scenes():
    rows = list(csv.DictReader(open(ROOT / "vision_dataset/manifests/unified_manifest.csv", encoding="utf-8")))
    cls = [r for r in rows if r["source_dataset"] == "classification"]
    return {c: len({r["scene_id"] for r in cls if r["cls_class"] == c}) for c in set(r["cls_class"] for r in cls)}


def fig04():
    cnt, _, _ = labels_counts()
    sc = scenes()
    order = sorted(cnt, key=lambda c: -cnt[c])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = [GREEN if c == "good" else BLUE for c in order]
    ax[0].bar(range(len(order)), [cnt[c] for c in order], color=cols)
    for i, c in enumerate(order):
        ax[0].text(i, cnt[c] + 4, cnt[c], ha="center", fontsize=8)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=35, ha="right")
    ax[0].set_title("Reviewed images per class (1,167)")
    s = [sc.get(c, 0) for c in order]
    ax[1].bar(range(len(order)), s, color=[GREEN if c == "good" else TEAL for c in order])
    for i, v in enumerate(s):
        ax[1].text(i, v + 0.4, v if v else "n/a", ha="center", fontsize=8)
    ax[1].axhline(50, color=RED, ls="--", lw=1)
    ax[1].text(len(order) - 0.5, 51, "commercial minimum ~50", color=RED, ha="right", fontsize=8)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=35, ha="right")
    ax[1].set_title("Distinct scenes (physical bottle passes) per class")
    fig.suptitle("Figure 4 - Classification dataset (project om_bottle)", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig04_dataset_distribution.png")


def fig05():
    base = ROOT / "projects/om_bottle/images"
    folders = [("+ve", "good")] + [(f"-ve/{d}", d) for d in ["Damaged Bottle", "Damaged Label", "Missing Cap",
                                                            "Missing Label", "Skewed Bottle", "Skewed Label",
                                                            "Tilt Cap", "Water Level"]]
    fig, axs = plt.subplots(1, 9, figsize=(14, 4))
    for ax, (f, name) in zip(axs, folders):
        imgs = sorted((base / f).glob("*.jpg")) + sorted((base / f).glob("*.png"))
        im = cv2.imread(str(imgs[len(imgs) // 2]))[:, :, ::-1]
        h, w = im.shape[:2]
        im = im[:, int(w * 0.28):int(w * 0.72)]            # centre strip: the bottle
        ax.imshow(im); ax.set_title(name, fontsize=8.5); ax.axis("off")
    fig.suptitle("Figure 5 - One training image per class (centre strip shown)", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig05_dataset_samples.png")


def fig06():
    fs = [("Training image (good)", "projects/om_bottle/images/+ve/snap_001.jpg"),
          ("Training image (tilt cap)", "projects/om_bottle/images/-ve/Tilt Cap/snap_145.jpg"),
          ("Production camera 2", "captures/TEST_20261007-085928/cam2_f1.png"),
          ("Production camera 3", "captures/TEST_20261007-085928/cam3_f1.png"),
          ("Line, empty belt", "projects/om_bottle/production/evidence/20261007")]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (t, p) in zip(axs, fs):
        pp = ROOT / p
        if pp.is_dir():
            pp = sorted(x for x in pp.glob("*.jpg") if "overlay" not in x.name)[0]
        im = cv2.imread(str(pp))[:, :, ::-1]
        ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle("Figure 6 - Domain gap: training images vs the production camera", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig06_domain_gap.png")


# ------------------------------------------------------------------ classifier figures
def fig07():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in cls_runs:
        h = r["m"]["history"]
        e = [x["epoch"] for x in h]
        ax[0].plot(e, [x["macro_f1"] for x in h], color=ARCH_COL[r["arch"]], label=NICE[r["arch"]], lw=1.6)
        ax[1].plot(e, [x["loss"] for x in h], color=ARCH_COL[r["arch"]], label=NICE[r["arch"]], lw=1.6)
        b = r["m"]["best_epoch"]
        ax[0].scatter([b], [h[b - 1]["macro_f1"]], color=ARCH_COL[r["arch"]], s=40, zorder=5, edgecolor="white")
    ax[0].set_title("Validation macro-F1 per epoch (dot = kept epoch)"); ax[0].set_xlabel("epoch"); ax[0].set_ylim(0.6, 1.02)
    ax[1].set_title("Training loss per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Figure 7 - Classifier training curves (2026-10-08 retrain, 25 epochs each)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "fig07_cls_training_curves.png")


def fig08():
    runs = cls_runs + [active]
    names = [NICE[r["arch"]] + ("\n(active, archived)" if r is active else "\n" + r["stamp"][-6:]) for r in runs]
    full = [r["t"]["test"]["macro_f1"] for r in runs]
    lf = [r["t"]["test_leak_free"]["macro_f1"] for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, full, 0.4, color=BLUE, label="full test set (246 / 238 images)")
    ax.bar(x + 0.2, lf, 0.4, color=TEAL, label="leak-free test set (226 / 218 images)")
    for i in range(len(runs)):
        ax.text(x[i] - 0.2, full[i] + 0.008, f"{full[i]:.3f}", ha="center", fontsize=7.5)
        ax.text(x[i] + 0.2, lf[i] + 0.008, f"{lf[i]:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0.6, 1.02); ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Figure 8 - Classifier held-out test macro-F1 (thresholds from validation)", loc="left")
    save(fig, "fig08_cls_test_f1.png")


def fig09():
    runs = cls_runs + [active]
    M = np.full((len(runs), len(CLASSES)), np.nan)
    for i, r in enumerate(runs):
        pd = r["t"]["test_leak_free"]["per_defect"]
        for j, c in enumerate(CLASSES):
            if c in pd and pd[c].get("n_pos"):
                M[i, j] = pd[c]["f1"]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.grid(False)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([NICE[r["arch"]] + (" (active)" if r is active else "") for r in runs])
    fig.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title("Figure 9 - Per-class F1 on the leak-free test set (n/a = no test positives)", loc="left")
    save(fig, "fig09_cls_perclass_heatmap.png")


def fig10():
    runs = cls_runs + [active]
    names = [NICE[r["arch"]] + (" (active)" if r is active else "") for r in runs]
    fp = [r["t"]["test_leak_free"]["false_pass"] for r in runs]
    fr = [r["t"]["test_leak_free"]["false_reject"] for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, fp, 0.4, color=RED, label="false PASS (defective bottle passed)")
    ax.bar(x + 0.2, fr, 0.4, color=AMBER, label="false REJECT (good bottle rejected)")
    for i in range(len(runs)):
        ax.text(x[i] - 0.2, fp[i] + 0.3, fp[i], ha="center", fontsize=8)
        ax.text(x[i] + 0.2, fr[i] + 0.3, fr[i], ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylabel("bottles")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure 10 - Bottle-level errors on the leak-free test set", loc="left")
    save(fig, "fig10_cls_errors.png")


def fig11():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in cls_runs:
        lat = r["t"].get("latency") or {}
        ms = lat.get("ms_mean") or r["m"]["infer_ms"]
        f1 = r["t"]["test_leak_free"]["macro_f1"]
        sz = r["m"]["model_size_mb"]
        ax.scatter(ms, f1, s=sz * 8, color=ARCH_COL[r["arch"]], alpha=0.75, edgecolor=INK)
        ax.annotate(f"{NICE[r['arch']]}\n{sz:.0f} MB", (ms, f1), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("inference time per image (ms, RTX 3050)"); ax.set_ylabel("leak-free test macro-F1")
    ax.set_title("Figure 11 - Accuracy vs speed (bubble = model size)", loc="left")
    save(fig, "fig11_cls_speed_accuracy.png")


def fig12():
    names, run, best, cols = [], [], [], []
    for r in cls_runs:
        names.append(NICE[r["arch"]]); run.append(r["m"]["epochs_run"]); best.append(r["m"]["best_epoch"])
        cols.append(BLUE)
    for tag, rr in (("YOLOv8n v3", "bench_yolov8n_v3"), ("YOLOv8s v3", "bench_yolov8s_v3")):
        rows = yolo_rows(rr)
        b = max(rows, key=lambda x: 0.1 * x["metrics/mAP50(B)"] + 0.9 * x["metrics/mAP50-95(B)"])
        names.append(tag); run.append(len(rows)); best.append(int(b["epoch"])); cols.append(GREEN)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, run, color=[c + "55" for c in cols], edgecolor=cols, label="epochs run")
    ax.bar(x, best, 0.45, color=cols, label="epoch whose weights were kept")
    for i in range(len(x)):
        ax.text(x[i], run[i] + 1, f"{best[i]} / {run[i]}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylabel("epochs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure 12 - How long each model trained, and which epoch was kept", loc="left")
    save(fig, "fig12_epochs.png")


def fig13():
    b0 = next(r for r in cls_runs if r["arch"] == "efficientnet_b0")
    pd = b0["m"]["per_defect"]
    cfg = jload(ROOT / "projects/om_bottle/config.json")["thresholds"]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, [pd[c]["threshold"] for c in CLASSES], 0.4, color=BLUE, label="chosen by training (B0 20261008)")
    ax.bar(x + 0.2, [min(cfg[c], 1.05) for c in CLASSES], 0.4, color=GREY, label="project config.json today")
    ax.axhline(0.9, color=GREEN, ls="--", lw=1); ax.text(7.5, 0.92, "proposed reject_min 0.90", ha="right", color=GREEN, fontsize=8)
    ax.axhline(0.1, color=RED, ls="--", lw=1); ax.text(7.5, 0.12, "proposed pass_max 0.10", ha="right", color=RED, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=25, ha="right"); ax.set_ylim(0, 1.1)
    ax.set_ylabel("decision threshold"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Figure 13 - Decision thresholds: many collapse to the 0.05 floor", loc="left")
    save(fig, "fig13_thresholds.png")


# ------------------------------------------------------------------ detector figures
def fig14():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, rr, col in (("YOLOv8n v3", "bench_yolov8n_v3", TEAL), ("YOLOv8s v3", "bench_yolov8s_v3", BLUE)):
        rows = yolo_rows(rr)
        e = [r["epoch"] for r in rows]
        ax[0].plot(e, [r["metrics/mAP50(B)"] for r in rows], color=col, label=tag)
        ax[1].plot(e, [r["metrics/mAP50-95(B)"] for r in rows], color=col, label=tag)
        ax[2].plot(e, [r["val/box_loss"] for r in rows], color=col, label=tag)
    ax[0].set_title("val mAP50"); ax[1].set_title("val mAP50-95"); ax[2].set_title("val box loss")
    ax[0].set_ylim(0.5, 1.0); ax[1].set_ylim(0.3, 0.9)
    for a in ax:
        a.set_xlabel("epoch")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 14 - Detector training curves (v3 split, 640 px)", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig14_yolo_curves.png")


def fig15():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    cl = ["bottle", "cap", "label"]
    k = 0
    for tag, f, col in (("YOLOv8n v3", "candidate_yolov8n_v3.json", TEAL), ("YOLOv8s v3", "candidate_yolov8s_v3.json", BLUE)):
        pc = jload(ROOT / "models/stage2_yolo" / f)["test"]["per_class"]
        for j, met in enumerate(["precision", "recall", "map50", "map50_95"]):
            vals = [pc[c][met] for c in cl]
            xs = np.arange(3) * 5 + j + k * 0.4
            ax.bar(xs, vals, 0.4, color=col, alpha=0.45 + 0.15 * j, label=f"{tag}" if j == 0 else None)
        k += 1
    ax.set_xticks(np.arange(3) * 5 + 1.7)
    ax.set_xticklabels(["bottle", "cap", "label"])
    for i in range(3):
        for j, met in enumerate(["P", "R", "mAP50", "mAP50-95"]):
            ax.text(i * 5 + j + 0.2, 0.32, met, rotation=90, fontsize=7, color="white", ha="center")
    ax.set_ylim(0.3, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Figure 15 - Detector per-class test metrics (52 test images, 6 scenes)", loc="left")
    save(fig, "fig15_yolo_perclass.png")


def copy_yolo_plots():
    src = HERE.parent / "19_Retrain_2026-10-08/yolo/yolov8s_v3"
    shutil.copy(src / "test_eval/PR_curve.png", FIG / "fig16_yolo_pr_curve.png")
    shutil.copy(src / "test_eval/confusion_matrix_normalized.png", FIG / "fig17_yolo_confusion.png")
    shutil.copy(src / "train/results.png", FIG / "fig16b_yolo_results.png")
    print("copied YOLO plots")


# ------------------------------------------------------------------ real camera + production
def fig19():
    r = jload(DATA / "real_camera_model_check.json")
    c = r["classifier"]
    names = [NICE[v["arch"]] for v in c.values()]
    flagged = [sum(1 for x in v["real"] if x["hits"]) for v in c.values()]
    avg = [np.mean([len(x["hits"]) for x in v["real"]]) for v in c.values()]
    n = len(next(iter(c.values()))["real"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    ax[0].bar(names, flagged, color=RED); ax[0].axhline(n, color=INK, ls=":", lw=1)
    ax[0].set_title(f"Real frames called defective (of {n})"); ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(names, avg, color=AMBER); ax[1].set_title("Defects reported per real frame")
    ax[1].tick_params(axis="x", rotation=30)
    b0 = c["20261008-081732"]
    cc = collections.Counter(h for x in b0["real"] for h in x["hits"])
    ax[2].barh(list(cc), list(cc.values()), color=PURPLE); ax[2].set_title("B0: which defect fired (frames)")
    fig.suptitle("Figure 19 - Classifiers on real production-camera frames (none of them a defective bottle)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "fig19_real_camera_classifiers.png")


def fig20():
    r = jload(DATA / "real_camera_model_check.json")["detector"]
    names = list(r)
    groups = {"no-bottle frames: no box": [], "bottle frames: bottle found": [], "cable-only frames: false box drawn": []}
    nob = lambda f: "NO_BOTTLE" in f or "TEST_20261005-214051" in f or "evidence" in f
    for nme in names:
        rows = r[nme]["real"]
        groups["no-bottle frames: no box"].append(
            sum(1 for x in rows if nob(x["frame"]) and not x["boxes"]) / max(1, sum(1 for x in rows if nob(x["frame"]))))
        bot = [x for x in rows if "TEST_20261007" in x["frame"] or "UNKNOWN" in x["frame"]]
        groups["bottle frames: bottle found"].append(
            sum(1 for x in bot if any(b[0] == "bottle" and b[1] >= 0.55 for b in x["boxes"])) / max(1, len(bot)))
        chk = [x for x in rows if "CHECK" in x["frame"]]
        groups["cable-only frames: false box drawn"].append(sum(1 for x in chk if x["boxes"]) / max(1, len(chk)))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (g, v) in enumerate(groups.items()):
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=[GREEN, BLUE, RED][i], label=g)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * 0.27, vi + 0.02, f"{vi:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.12); ax.set_ylabel("share of frames")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Figure 20 - Detectors on real production-camera frames (bottle conf >= 0.55)", loc="left")
    save(fig, "fig20_real_camera_detectors.png")


def fig21():
    db = sqlite3.connect(ROOT / "projects/om_bottle/production/production.db")
    groups = collections.Counter()
    finals = collections.Counter()
    for final, data in db.execute("select final, data from inspections"):
        finals[final] += 1
        reason = (json.loads(data or "{}").get("reason") or "")
        g = ("Not inspected: M2 still ON (previous bottle)" if "M2 already ON" in reason else
             "Not inspected: M1 reject cycle masks trigger" if "M1 ON" in reason else
             "Line halted (3 FAULTs in a row)" if "line halted" in reason else
             "No bottle found by detector" if "no bottle found" in reason else
             "Stale trigger at start" if "stale trigger" in reason else
             "REJECT (several defects)" if final == "REJECT" else "Camera fault / other")
        groups[g] += 1
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 2.0], "wspace": 0.95})
    ax[0].bar(["PASS", "REJECT", "FAULT"], [finals.get("PASS", 0), finals.get("REJECT", 0), finals.get("FAULT", 0)],
              color=[GREEN, RED, AMBER])
    for i, k in enumerate(["PASS", "REJECT", "FAULT"]):
        ax[0].text(i, finals.get(k, 0) + 2, finals.get(k, 0), ha="center")
    ax[0].set_title(f"Final result ({sum(finals.values())} bottles)")
    it = groups.most_common()
    ax[1].barh([k for k, _ in it][::-1], [v for _, v in it][::-1], color=BLUE)
    for i, (k, v) in enumerate(it[::-1]):
        ax[1].text(v + 1, i, v, va="center", fontsize=8)
    ax[1].set_title("Why")
    fig.suptitle("Figure 21 - Real-line trial 2026-10-06/07 (production.db)", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig21_production_trial.png")


def fig22():
    items = [("Clock tick (time.monotonic, Py3.8)", 0, 15.6, "measured"),
             ("Frame stamp: exposure > read() return", 40, 150, "estimate, typical UVC"),
             ("Trigger poll period + scan", 10, 70, "from settings"),
             ("Trigger: full status read at 9600 baud", 100, 300, "estimate, 6+ transactions"),
             ("Target after Phase 1", 0, 20, "goal")]
    fig, ax = plt.subplots(figsize=(10, 3.3))
    for i, (n, lo, hi, src) in enumerate(items[::-1]):
        col = GREEN if "Target" in n else BLUE if src == "measured" else AMBER
        ax.barh(i, hi - lo, left=lo, color=col, alpha=0.85)
        ax.text(hi + 4, i, f"{lo:g}-{hi:g} ms ({src})", va="center", fontsize=8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=8.5)
    ax.set_xlim(0, 420); ax.set_xlabel("milliseconds")
    ax.set_title("Figure 22 - Timestamp error budget of the camera / trigger chain", loc="left")
    save(fig, "fig22_timestamp_budget.png")


def overlays():
    for src, dst in (("captures/TEST_20261007-085928/cam3_f1_overlay.jpg", "fig23_app_overlay_cam3.jpg"),
                     ("captures/TEST_20261007-085928/cam2_f1_overlay.jpg", "fig24_app_overlay_cam2.jpg"),
                     ("captures/GT_NOCAP_20261003-134650/img_328_f1_overlay.jpg", "fig25_missing_cap_test.jpg"),
                     ("FINAL_YEAR_BLACKBOOK/16_System_Analysis/fig_domain_shift.png", "fig26_domain_shift_earlier.png")):
        p = ROOT / src
        if p.exists():
            shutil.copy(p, FIG / dst)
            print("copied", dst)


if __name__ == "__main__":
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13,
              fig14, fig15, copy_yolo_plots, fig19, fig20, fig21, fig22, overlays):
        f()
