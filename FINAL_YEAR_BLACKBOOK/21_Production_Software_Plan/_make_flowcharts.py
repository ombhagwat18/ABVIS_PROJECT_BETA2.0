"""Draws the three process flowcharts of the production-ready software plan (read-only for the project).

    python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_make_flowcharts.py

fig_plan_01_roadmap.png       overall plan: software phases -> freeze gate -> hardware -> validation -> release
fig_plan_02_bottle_cycle.png  target per-bottle production flow with every fault branch
fig_plan_03_model_lifecycle.png  data and model lifecycle (capture -> review -> train -> gates -> active -> monitor)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Patch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# same palette as 20_Final_Report/_make_figures.py
INK, DIM = "#1f2933", "#616e7c"
BLUE, GREEN, RED, AMBER, GREY = "#2f6fb0", "#2e8b57", "#c0392b", "#d68910", "#95a5a6"
F_DONE, F_SOFT, F_HW, F_FUT, F_GATE, F_BAD, F_TERM = (
    "#e8f6ef", "#e3edf8", "#fdf2e3", "#eef1f4", "#fff8d6", "#fbe9e7", "#dfe6ee")
plt.rcParams["font.family"] = "Calibri"


class N:
    """A flowchart node; anchors t/b/l/r are the middle of each side (or the diamond's vertices)."""

    def __init__(self, ax, kind, x, y, w, h, text, fc, fs=9, bold=False, ec=None, ls="-"):
        self.x, self.y, self.w, self.h = x, y, w, h
        ec = ec or INK
        if kind == "dia":
            ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                                 fc=fc, ec=ec, lw=1.2))
        else:
            r = 0.35 if kind == "term" else 0.08
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=1.2, ls=ls,
                                        boxstyle=f"round,pad=0.02,rounding_size={r}"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25,
                fontweight="bold" if bold else "normal")

    t = property(lambda s: (s.x, s.y + s.h / 2))
    b = property(lambda s: (s.x, s.y - s.h / 2))
    l = property(lambda s: (s.x - s.w / 2, s.y))
    r = property(lambda s: (s.x + s.w / 2, s.y))


def link(ax, *pts, label="", col=INK, lpos=0, dx=0.12, dy=0.1):
    """Orthogonal connector through pts; arrow head on the last point; label near point lpos."""
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=col, lw=1.3, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3,
                                                              shrinkA=0, shrinkB=0))
    if label:
        x, y = pts[lpos]
        ax.text(x + dx, y + dy, label, fontsize=8.5, color=col, fontweight="bold")


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. overall plan
def fig_roadmap():
    fig, ax = plt.subplots(figsize=(10.3, 15.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 16.6); ax.axis("off")
    ax.legend(handles=[Patch(fc=F_DONE, ec=INK, label="done (software, self-tested)"),
                       Patch(fc=F_SOFT, ec=INK, label="software work now (no hardware)"),
                       Patch(fc=F_HW, ec=INK, label="needs the physical machine"),
                       Patch(fc=F_GATE, ec=INK, label="decision gate"),
                       Patch(fc=F_FUT, ec=INK, label="after v1.0")],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5, frameon=False)
    now = N(ax, "term", 5.5, 15.25, 8.4, 0.95,
            "Today (14 Oct 2026): software core built and self-tested (22/22, fakes)\n"
            "one real-line trial showed the one-bottle PLC handshake and the camera domain gap", F_DONE, bold=True)
    a = N(ax, "rect", 5.5, 13.85, 8.4, 1.15,
          "Phase A - Safe, consistent software baseline\n"
          "safe production settings (FAULT = REJECT, no live threshold shift), COM-port pinning,\n"
          "archive rollback, data paths, every self-check green, status documents corrected", F_SOFT, bold=False)
    b = N(ax, "rect", 2.05, 11.55, 3.3, 2.25,
          "Phase B - Line protocol v2\n(simulator only)\n\nPLC bottle queue: answer each\nbottle at once, PLC fires Y0\n"
          "heartbeat watchdog\nshort inspection window\ntarget 60 bottles/min", F_SOFT)
    c = N(ax, "rect", 5.5, 11.55, 3.3, 2.25,
          "Phase C - AI pipeline\n\nreview bootstrapped defect\nboxes (detector v4 classes)\nsplit by physical bottle\n"
          "YOLOv8s v4 candidate\ndetector-guided crop\ntarget-based thresholds", F_SOFT)
    d = N(ax, "rect", 8.95, 11.55, 3.3, 2.25,
          "Phase D - Operations\n\nExcel workbook from SQLite\ncommissioning report export\n"
          "photo-eye / camera latency\ntest tool\nrecovery tests, operator guide", F_SOFT)
    g1 = N(ax, "dia", 5.5, 9.15, 5.6, 1.5,
           "Software freeze gate\nall checks green, protocol v2 proven in the\nsimulator, documents honest?", F_GATE, fs=8.8)
    e = N(ax, "rect", 5.5, 7.5, 8.4, 1.25,
          "Phase E - Hardware commissioning (later, staged, read-only first)\n"
          "PLC link > loaded ladder vs saved .isp > T0/T1 > guarded PASS / REJECT handshake >\n"
          "cameras on final USB + lighting > measured belt speed and distances > end-to-end bottles", F_HW)
    g2 = N(ax, "dia", 5.5, 5.85, 5.6, 1.45,
           "One good bottle PASS and one defective\nbottle REJECT, filmed; PLC, database and\nbottle counts reconcile?",
           F_GATE, fs=8.8)
    f = N(ax, "rect", 5.5, 4.3, 8.4, 1.05,
          "Phase F - Production-camera data, retraining and validation\n"
          ">= 2,000 good + >= 300 per defect on the EMEETs, split by bottle; candidate > validated > approved > active",
          F_HW)
    g3 = N(ax, "dia", 5.5, 2.85, 5.6, 1.35,
           "Acceptance gates met?\nescaped defects <= 0.1 %, false rejects <= 1 %,\nunsure <= 2 %, repeatability",
           F_GATE, fs=8.8)
    rel = N(ax, "term", 5.5, 1.5, 8.4, 0.8,
            "v1.0  Production-ready bottle inspection system", "#c9ebd7", bold=True, fs=10.5)
    g = N(ax, "rect", 5.5, 0.4, 8.4, 0.65,
          "Phase G - Generalise the proven parts into the universal inspection platform", F_FUT)

    link(ax, now.b, a.t)
    for n in (b, c, d):
        link(ax, (n.x, a.b[1]), n.t)
    link(ax, c.b, g1.t)
    for n in (b, d):
        ax.plot([n.x, n.x, c.x], [n.b[1], 10.05, 10.05], color=INK, lw=1.3)
    link(ax, g1.b, e.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, g1.l, (0.12, 9.15), (0.12, 13.85), a.l, label="No: fix", lpos=0, dx=-1.1)
    link(ax, e.b, g2.t)
    link(ax, g2.b, f.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, g2.r, (10.85, 5.85), (10.85, 7.5), e.r, label="No: coordinated\nladder + software fix",
         lpos=1, dx=-2.3, dy=-0.75, col=RED)
    link(ax, f.b, g3.t)
    link(ax, g3.b, rel.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, g3.l, (0.25, 2.85), (0.25, 4.3), f.l, label="No: more\nproduction data", lpos=0, dx=-2.3, dy=-0.75,
         col=RED)
    link(ax, rel.b, g.t)
    save(fig, "fig_plan_01_roadmap.png")


# ------------------------------------------------------------------ 2. per-bottle production flow
def fig_bottle_cycle():
    fig, ax = plt.subplots(figsize=(11.5, 14.6))
    ax.set_xlim(0, 14); ax.set_ylim(-0.1, 17.7); ax.axis("off")
    # start-up row
    s1 = N(ax, "term", 1.45, 15.2, 2.5, 0.85, "Operator presses\nSTART", F_TERM, bold=True)
    s2 = N(ax, "dia", 4.75, 15.2, 3.0, 1.55, "Readiness\nchecklist OK?", F_GATE)
    s3 = N(ax, "rect", 8.05, 15.2, 2.7, 1.15, "Load models (background)\nopen line cameras\nwarm-up", F_SOFT)
    s4 = N(ax, "dia", 11.45, 15.2, 3.0, 1.55, "Fresh frames\nfrom every\ncamera?", F_GATE)
    n2 = N(ax, "rect", 4.75, 16.95, 3.6, 0.7, "NOT_READY: show the blocking reason", F_BAD, fs=8.5)
    n4 = N(ax, "rect", 11.45, 16.95, 3.6, 0.7, "CAMERA_DISCONNECTED: no start", F_BAD, fs=8.5)
    link(ax, s1.r, s2.l)
    link(ax, s2.r, s3.l, label="Yes", lpos=0, dx=0.05)
    link(ax, s3.r, s4.l)
    link(ax, s2.t, n2.b, label="No", lpos=0, dx=0.1, dy=0.05)
    link(ax, s4.t, n4.b, label="No", lpos=0, dx=0.1, dy=0.05)

    X = 5.6
    m1 = N(ax, "rect", X, 13.05, 5.0, 0.95, "RUNNING: wait for the bottle trigger\n(photo-eye X0 -> PLC ladder sets M2)",
           F_SOFT, bold=True)
    link(ax, s4.b, (11.45, 13.95), (X, 13.95), m1.t, label="Yes", lpos=0, dx=0.1, dy=-0.3)
    m2 = N(ax, "rect", X, 11.75, 5.0, 0.95,
           "New inspection_id + trigger time (+/- uncertainty)\nopen each camera's frame window", F_SOFT)
    m3 = N(ax, "rect", X, 10.45, 5.0, 0.95,
           "Frames captured after the trigger ->\nAI worker: classifier + YOLO detector", F_SOFT)
    m4 = N(ax, "dia", X, 8.85, 5.0, 1.75,
           "Evidence valid?\nfresh frames, bottle in >= 60 % of\nframes, association OK, models OK", F_GATE, fs=8.6)
    m5 = N(ax, "rect", X, 7.2, 5.0, 0.95,
           "decision.py: recipe rules, three-zone thresholds,\nframe vote, camera fusion FAULT > REJECT > PASS", F_SOFT)
    m6 = N(ax, "dia", X, 5.75, 3.3, 1.35, "Final verdict?", F_GATE, bold=True)
    link(ax, m1.b, m2.t); link(ax, m2.b, m3.t); link(ax, m3.b, m4.t)
    link(ax, m4.b, m5.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, m5.b, m6.t)

    p1 = N(ax, "rect", 2.0, 4.25, 2.6, 0.85, "PASS:\nwrite M0 now", F_DONE, bold=True)
    r1 = N(ax, "rect", X, 4.25, 3.6, 0.85, "REJECT: schedule M1 at\ntrigger + travel - T0", F_BAD, bold=True)
    f1 = N(ax, "rect", 11.25, 5.75, 3.6, 1.15, "FAULT / UNSURE\nfault_action = REJECT\n(never a silent PASS)", F_BAD,
           bold=True)
    link(ax, m6.l, (2.0, 5.75), p1.t, label="PASS", lpos=0, dx=-0.9)
    link(ax, m6.b, r1.t, label="REJECT", lpos=0, dy=-0.27)
    link(ax, m6.r, f1.l, label="FAULT", lpos=0, dx=0.1)
    link(ax, m4.r, (11.25, 8.85), f1.t, label="No", lpos=0, dx=0.1)
    link(ax, f1.b, (11.25, 4.25), r1.r)

    r2 = N(ax, "dia", X, 2.85, 3.4, 1.25, "On time for\nthis bottle?", F_GATE)
    late = N(ax, "rect", 11.25, 2.85, 3.6, 1.0, "Do not fire late: FAULT +\nmissed-reject alarm,\nremove the bottle by hand",
             F_BAD, fs=8.5)
    ack = N(ax, "dia", X, 1.45, 3.4, 1.0, "PLC\nacknowledged?", F_GATE)
    nack = N(ax, "rect", 11.25, 1.45, 3.6, 0.85, "NOT_ACKED / ACK_LOST:\nFAULT alarm, never retried", F_BAD, fs=8.5)
    rec = N(ax, "rect", X, 0.35, 6.4, 0.6, "Record: SQLite + CSV + evidence image + logs -> HMI counters", F_DONE,
            bold=True)
    link(ax, r1.b, r2.t)
    link(ax, r2.r, late.l, label="No", lpos=0, dx=0.1)
    link(ax, r2.b, ack.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, p1.b, (2.0, 1.45), ack.l)
    link(ax, ack.r, nack.l, label="No", lpos=0, dx=0.1)
    link(ax, ack.b, rec.t, label="Yes", lpos=0, dy=-0.22)
    link(ax, late.r, (13.3, 2.85), (13.3, 0.35), rec.r)
    ax.plot([nack.b[0], nack.b[0]], [nack.b[1], 0.35], color=INK, lw=1.3)
    # loop back to the next bottle
    link(ax, rec.l, (0.2, 0.35), (0.2, 13.05), m1.l, label="next\nbottle", lpos=1, dx=0.12, dy=5.6, col=BLUE)
    N(ax, "rect", 1.62, 9.6, 2.45, 3.2,
      "Safety latch (software)\n\nSTOP, E-stop input or\n3 FAULTs in a row\n-> HALT: no M0/M1 sent,\n"
      "every trigger = FAULT,\nRESET FAULT to resume\n\nThe hardwired E-stop\ncuts power by itself", "#ffffff",
      fs=8.3, ec=RED, ls="--")
    save(fig, "fig_plan_02_bottle_cycle.png")


# ------------------------------------------------------------------ 3. model lifecycle
def fig_lifecycle():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 14.4); ax.set_ylim(-0.4, 5.6); ax.axis("off")
    t1 = N(ax, "rect", 1.9, 4.7, 3.2, 1.25, "Capture on the production\ncamera (auto-collect,\ninbox, reviewed = 0)", F_HW)
    t2 = N(ax, "rect", 5.4, 4.7, 3.0, 1.25, "Human review\nlabels + boxes accepted\n(suggestions never trusted)",
           F_SOFT)
    t3 = N(ax, "rect", 8.9, 4.7, 3.0, 1.25, "Leakage-safe split\nby physical bottle / scene\ntrain | val | test", F_SOFT)
    t4 = N(ax, "rect", 12.4, 4.7, 3.0, 1.25, "Train CANDIDATE\none job at a time on GPU\nnever auto-activated", F_SOFT)
    b1 = N(ax, "rect", 12.4, 1.75, 3.0, 1.25, "VALIDATED\nheld-out test + real-\ncamera record + shortcut check",
           F_SOFT)
    b2 = N(ax, "dia", 8.9, 1.75, 3.0, 1.6, "Acceptance\ngates met?", F_GATE, bold=True)
    b3 = N(ax, "rect", 5.4, 1.75, 3.0, 1.25, "APPROVED -> ACTIVE\nprevious model archived,\nrollback ready", F_DONE,
           bold=True)
    b4 = N(ax, "rect", 1.9, 1.75, 3.2, 1.25, "Production monitoring\nunsure / fault rate,\noperator corrections,\nhard examples",
           F_HW)
    rej = N(ax, "rect", 8.9, -0.05, 3.0, 0.6, "REJECTED (kept as evidence)", F_BAD, fs=8.5)
    link(ax, t1.r, t2.l); link(ax, t2.r, t3.l); link(ax, t3.r, t4.l)
    link(ax, t4.b, b1.t)
    link(ax, b1.l, b2.r)
    link(ax, b2.l, b3.r, label="Yes", lpos=0, dx=-0.48, dy=0.12)
    link(ax, b2.b, rej.t, label="No", lpos=0, dx=0.1, dy=-0.3)
    link(ax, b3.l, b4.r)
    link(ax, b4.t, t1.b, label="new dataset version", lpos=0, dx=0.12, dy=0.5, col=BLUE)
    save(fig, "fig_plan_03_model_lifecycle.png")


if __name__ == "__main__":
    fig_roadmap()
    fig_bottle_cycle()
    fig_lifecycle()
