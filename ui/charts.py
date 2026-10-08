"""Small chart drawing straight onto a Tk canvas.

No matplotlib. Three chart types over a few hundred points each does not
justify a 40 MB dependency, a second event loop and a figure-to-image round
trip on every redraw -- and matplotlib's Tk backend fights the theme.

Everything here draws in the theme.py palette and takes explicit colours, so
the widgets and the plots cannot drift apart.
"""
from __future__ import annotations

import math

from ui import theme

INK = theme.INK
DIM = theme.DIM
LINE = theme.LINE
BG = theme.PANEL                # charts sit on panels
BLUE = theme.ACC
BLUES = theme.SERIES
RED = theme.REJECT
GREEN = theme.PASS
AMBER = theme.FAULT

PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 14, 30


def nice_ticks(lo: float, hi: float, want: int = 5) -> list[float]:
    """Round tick values covering [lo, hi].

    Axis labels like 0.0237 are noise; a reader wants 0.02, 0.04. Snap the step
    to 1, 2 or 5 times a power of ten.
    """
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, want)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = next((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), 10 * mag)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5 and len(out) < 24:
        out.append(round(v, 10))
        v += step
    return out


def _plot_box(w: int, h: int):
    return PAD_L, PAD_T, max(PAD_L + 1, w - PAD_R), max(PAD_T + 1, h - PAD_B)


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}"


def line_chart(canvas, series: dict[str, list[float]], x_values=None,
               y_lo=None, y_hi=None, title="", legend=True, font=("Segoe UI", 9)):
    """series: {label: [y, ...]}. Draws axes, grid, one polyline per series."""
    canvas.delete("all")
    w = int(canvas.winfo_width()) or int(canvas["width"])
    h = int(canvas.winfo_height()) or int(canvas["height"])
    x0, y0, x1, y1 = _plot_box(w, h)
    series = {k: v for k, v in series.items() if v}
    if not series:
        canvas.create_text(w // 2, h // 2, text="no data yet", fill=DIM, font=font)
        return

    n = max(len(v) for v in series.values())
    xs = x_values or list(range(1, n + 1))
    flat = [v for vals in series.values() for v in vals]
    lo = min(flat) if y_lo is None else y_lo
    hi = max(flat) if y_hi is None else y_hi
    if hi == lo:
        hi = lo + 1.0
    ticks = nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    def px(i):
        return x0 + (x1 - x0) * (i / max(1, n - 1)) if n > 1 else (x0 + x1) / 2

    def py(v):
        return y1 - (y1 - y0) * ((v - lo) / (hi - lo))

    for t in ticks:
        y = py(t)
        if y0 - 1 <= y <= y1 + 1:
            canvas.create_line(x0, y, x1, y, fill=LINE)
            canvas.create_text(x0 - 6, y, text=_fmt(t), anchor="e", fill=DIM, font=font)
    canvas.create_line(x0, y0, x0, y1, fill=DIM)
    canvas.create_line(x0, y1, x1, y1, fill=DIM)

    step = max(1, n // 8)
    for i in range(0, n, step):
        canvas.create_text(px(i), y1 + 12, text=str(xs[min(i, len(xs) - 1)]),
                           fill=DIM, font=font)

    for k, (label, vals) in enumerate(series.items()):
        colour = BLUES[k % len(BLUES)]
        pts = []
        for i, v in enumerate(vals):
            pts += [px(i), py(v)]
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=colour, width=2, smooth=False)
        elif len(pts) == 2:
            canvas.create_oval(pts[0] - 3, pts[1] - 3, pts[0] + 3, pts[1] + 3,
                               fill=colour, outline="")
        if legend:
            ly = y0 + 4 + k * 14
            canvas.create_line(x1 - 92, ly, x1 - 76, ly, fill=colour, width=3)
            canvas.create_text(x1 - 72, ly, text=label, anchor="w", fill=INK, font=font)
    if title:
        canvas.create_text(x0, 4, text=title, anchor="nw", fill=INK, font=font)


def bar_chart(canvas, labels: list[str], values: list[float], colours=None,
              y_hi=None, font=("Segoe UI", 9)):
    canvas.delete("all")
    w = int(canvas.winfo_width()) or int(canvas["width"])
    h = int(canvas.winfo_height()) or int(canvas["height"])
    x0, y0, x1, y1 = _plot_box(w, h)
    if not values:
        canvas.create_text(w // 2, h // 2, text="no data yet", fill=DIM, font=font)
        return
    hi = y_hi if y_hi is not None else max(max(values), 1e-9)
    for t in nice_ticks(0, hi):
        y = y1 - (y1 - y0) * (t / hi)
        if y0 - 1 <= y <= y1 + 1:
            canvas.create_line(x0, y, x1, y, fill=LINE)
            canvas.create_text(x0 - 6, y, text=_fmt(t), anchor="e", fill=DIM, font=font)
    canvas.create_line(x0, y1, x1, y1, fill=DIM)

    slot = (x1 - x0) / max(1, len(values))
    # Fit the label to its slot instead of a fixed character count: seven
    # defect names at ~65 px each ran into one another and read as one long
    # word. Below about four characters, stagger every other label instead.
    maxch = max(3, int(slot / (font[1] * 0.72)))
    stagger = maxch < 6
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = x0 + slot * (i + 0.5)
        bw = min(46, slot * 0.62)
        top = y1 - (y1 - y0) * (max(0.0, v) / hi)
        canvas.create_rectangle(cx - bw / 2, top, cx + bw / 2, y1,
                                fill=(colours[i] if colours else BLUE), outline="")
        canvas.create_text(cx, top - 8, text=_fmt(v), fill=INK, font=font)
        text = lab if len(lab) <= maxch else lab[:max(1, maxch - 1)] + "…"
        canvas.create_text(cx, y1 + 12 + (10 if stagger and i % 2 else 0),
                           text=text if not stagger else lab[:maxch],
                           fill=DIM, font=font)


def confusion(canvas, tp: int, fp: int, fn: int, tn: int, font=("Segoe UI", 10)):
    """One defect's 2x2 confusion matrix.

    Multi-label means there is no single N-by-N matrix: each defect is its own
    yes/no question, and one image can be a true positive for water_level and a
    false negative for tilt_cap at the same time. Drawing a combined matrix
    would have to invent a single label per image and would be a picture of a
    model nobody trained.
    """
    canvas.delete("all")
    w = int(canvas.winfo_width()) or int(canvas["width"])
    h = int(canvas.winfo_height()) or int(canvas["height"])
    left, top = 92, 34
    cw, ch = max(40, (w - left - 12) / 2), max(30, (h - top - 26) / 2)
    cells = [("TN", tn, theme.PANEL_2, INK), ("FP", fp, theme.REJECT_SOFT, RED),
             ("FN", fn, theme.REJECT_SOFT, RED), ("TP", tp, theme.PASS_SOFT, GREEN)]
    canvas.create_text(left + cw, 12, text="predicted", fill=DIM, font=font)
    for i, (name, v, bg, fg) in enumerate(cells):
        r, c = divmod(i, 2)
        x, y = left + c * cw, top + r * ch
        canvas.create_rectangle(x, y, x + cw, y + ch, fill=bg, outline=BG, width=2)
        canvas.create_text(x + cw / 2, y + ch / 2 - 7, text=str(v), fill=fg,
                           font=(font[0], font[1] + 6, "bold"))
        canvas.create_text(x + cw / 2, y + ch / 2 + 13, text=name, fill=DIM, font=font)
    for c, t in enumerate(("no", "yes")):
        canvas.create_text(left + cw * (c + 0.5), top - 9, text=t, fill=DIM, font=font)
    for r, t in enumerate(("actual no", "actual yes")):
        canvas.create_text(left - 8, top + ch * (r + 0.5), text=t, anchor="e",
                           fill=DIM, font=font)
    miss = fn / (tp + fn) if (tp + fn) else 0.0
    canvas.create_text(left, top + 2 * ch + 12, anchor="w", fill=RED if fn else DIM,
                       font=font,
                       text=f"{fn} of {tp + fn} real defects missed ({miss:.0%})")


def demo():
    """Self-check on the tick maths -- the only part that can be quietly wrong."""
    t = nice_ticks(0, 1)
    assert t[0] <= 0 and t[-1] >= 1 and len(t) <= 12, t
    assert all(b > a for a, b in zip(t, t[1:])), t
    for lo, hi in ((0, 0.0237), (-5, 5), (0, 1e6), (3, 3), (0.1, 0.10001)):
        ts = nice_ticks(lo, hi)
        assert ts[0] <= lo and ts[-1] >= hi, (lo, hi, ts)
        assert 2 <= len(ts) <= 24, (lo, hi, ts)
    # steps must be a round 1/2/5 x power of ten, not an arbitrary fraction
    step = nice_ticks(0, 100)[1] - nice_ticks(0, 100)[0]
    assert round(step / 10 ** math.floor(math.log10(step)), 6) in (1.0, 2.0, 5.0), step
    # degenerate input must not hang or divide by zero
    assert nice_ticks(float("nan"), 1) == [0.0, 1.0]
    assert nice_ticks(5, 5)[0] <= 5
    assert _fmt(3.0) == "3" and _fmt(0.25) == "0.25"
    print(f"ok  ticks round to 1/2/5, {len(t)} on [0,1], degenerate ranges safe")


if __name__ == "__main__":
    demo()
