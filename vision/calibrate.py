"""Measure where the bottle actually sits, and propose an ROI.

The background is flat and unlit; the bottle is all edges (cap rim, label print,
ribs, meniscus). So edge energy separates them far more reliably than brightness
would -- the bottle body is transparent and nearly as dark as the backdrop, which
is exactly where a plain Otsu threshold falls over.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from vision import dataset as D

PAD = 0.06        # margin so a skewed bottle still fits
WORK = 400        # analyse at this width; full res just adds sensor noise
RATIO = 0.15      # edge energy above baseline + 15% of the peak counts as bottle


def _span(v: np.ndarray) -> tuple[int, int] | None:
    """First and last index where the profile rises clearly above its own floor.

    The floor is not zero: the backdrop has a lighting gradient and the table
    edge draws a line across every row. So the threshold is relative to each
    profile's own baseline, never an absolute edge value.
    """
    k = max(3, len(v) // 50)
    v = np.convolve(v, np.ones(k) / k, "same")     # smooth over gaps (label seams, ribs)
    base = float(np.percentile(v, 10))
    peak = float(v.max())
    if peak - base < 1e-6:
        return None
    on = np.flatnonzero(v > base + RATIO * (peak - base))
    if len(on) < 3:
        return None
    return int(on[0]), int(on[-1]) + 1


def bbox(img: np.ndarray) -> tuple[int, int, int, int] | None:
    H, W = img.shape[:2]
    s = WORK / W
    small = cv2.resize(img, (WORK, max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    x = _span(e.sum(0))
    y = _span(e.sum(1))
    if not x or not y:
        return None
    return (round(x[0] / s), round(y[0] / s), round((x[1] - x[0]) / s), round((y[1] - y[0]) / s))


def run(sample: int = 60) -> list[int]:
    paths = D.scan_images()
    if not paths:
        raise SystemExit(f"no images in project {D.PROJECT!r} -- upload some first")
    step = max(1, len(paths) // sample)
    boxes = []
    for p in paths[::step][:sample]:
        img = D.imread(D.IMAGE_ROOT / p)
        if img is None:
            continue
        b = bbox(img)
        if b:
            boxes.append(b)
            print(f"  {p:<44} {b}")
    if not boxes:
        raise SystemExit("could not find a bottle in any sample")

    a = np.array(boxes)
    H, W = img.shape[:2]
    # union of the middle 95% -- one bad frame must not blow the ROI out to full size
    x0 = np.percentile(a[:, 0], 2.5)
    y0 = np.percentile(a[:, 1], 2.5)
    x1 = np.percentile(a[:, 0] + a[:, 2], 97.5)
    y1 = np.percentile(a[:, 1] + a[:, 3], 97.5)
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    roi = [int(max(0, x0 - px)), int(max(0, y0 - py)),
           int(min(W, x1 + px) - max(0, x0 - px)),
           int(min(H, y1 + py) - max(0, y0 - py))]

    cfg = D.load_config()
    cfg["roi"] = roi
    cfg["roi_frame"] = [int(W), int(H)]   # so the box can rescale to other frame sizes
    D.save_config(cfg)
    print(f"\nframe   {W}x{H}")
    print(f"roi     {roi}   ({roi[2]}x{roi[3]}, {100 * roi[2] * roi[3] / (W * H):.0f}% of frame)")
    print(f"saved to {D.CONFIG_JSON}")
    print("check it on the Data health tab -> Preview crop before training")
    return roi


def demo():
    """A textured block on a gradient field must be found, gradient and all --
    the gradient is the thing the old absolute threshold tripped over."""
    rng = np.random.default_rng(0)
    img = np.zeros((900, 1200, 3), np.uint8)
    img[:] = np.linspace(30, 70, 900, dtype=np.uint8)[:, None, None]   # backdrop gradient
    img[:, :, :] += rng.integers(0, 4, img.shape, dtype=np.uint8)      # sensor noise
    img[150:800, 500:700] = rng.integers(0, 255, (650, 200, 3), dtype=np.uint8)
    x, y, w, h = bbox(img)
    assert 440 <= x <= 540 and 100 <= y <= 200, (x, y)
    assert 160 <= w <= 300 and 580 <= h <= 740, (w, h)
    assert bbox(np.full((300, 400, 3), 40, np.uint8)) is None, "flat field has no bottle"
    print("ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else run()
