"""Experiment A: one classification project per camera, each with its own fixed ROI (no cropping step at run time).

    python -m tools.camera_roi build     # projects enclosure_cama / enclosure_camb from enclosure_cls (hard links, labels filtered) + ROI
    python -m tools.camera_roi --selftest

The ROI of a camera = where that camera ever sees the bottle: the 2nd..98th percentile of the bottle boxes the auto-labeller drew
(only a STATISTIC, no label is taken from it) plus a margin, written to config.json as roi / roi_frame so dataset.crop rescales it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision import dataset as D                                           # noqa: E402

SRC = "enclosure_cls"
FRAME = (3840, 2160)
INPUT_WH, CACHE_WH = [416, 480], [448, 512]
MARGIN = 0.04


def roi_from_boxes(boxes, frame=FRAME, margin=MARGIN, lo=2, hi=98):
    """boxes = [(x0,y0,x1,y1) normalized] -> [x, y, w, h] in pixels: percentile union + margin, clamped to the frame."""
    b = np.array(boxes, dtype=float)
    x0, y0 = np.percentile(b[:, 0], lo), np.percentile(b[:, 1], lo)
    x1, y1 = np.percentile(b[:, 2], hi), np.percentile(b[:, 3], hi)
    x0, y0, x1, y1 = max(0.0, x0 - margin), max(0.0, y0 - margin), min(1.0, x1 + margin), min(1.0, y1 + margin)
    W, H = frame
    return [int(x0 * W), int(y0 * H), int((x1 - x0) * W), int((y1 - y0) * H)]


def bottle_boxes_by_camera():
    ann = json.loads((ROOT / "projects" / "enclosure_seg" / "annotations_autoseed_backup.json").read_text())
    cam = json.loads((ROOT / "projects" / SRC / "prep" / "camera_final.json").read_text())
    cam_by_name = {k.split("/")[1]: v for k, v in cam.items()}
    out = {"A": [], "B": []}
    for rel, e in ann["images"].items():
        c = cam_by_name.get(Path(rel).name)
        poly = next((p for p in e.get("polygons") or [] if p["cls"] == "bottle" and p.get("score", 1) >= 0.8), None)
        if c in out and poly:
            xs, ys = poly["points"][0::2], poly["points"][1::2]
            out[c].append((min(xs), min(ys), max(xs), max(ys)))
    return out, cam_by_name


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build():
    D.use_project(SRC)
    src_dir, src_images = D.PROJECT_DIR, D.IMAGE_ROOT
    boxes, cam_by_name = bottle_boxes_by_camera()
    with open(src_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    made = {}
    for c in ("A", "B"):
        name = D.create_project(f"Enclosure Cam{c}", "classification")
        pdir = ROOT / "projects" / name
        keep = [r for r in body if cam_by_name.get(Path(r[0]).name) == c]
        for r in keep:
            link(src_images / r[0], pdir / "images" / r[0])
        for d in head[1:-1]:
            (pdir / "images" / "-ve" / d).mkdir(parents=True, exist_ok=True)
        with open(pdir / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(keep)
        roi = roi_from_boxes(boxes[c])
        cfg = {"roi": roi, "roi_frame": list(FRAME), "input_wh": INPUT_WH, "cache_wh": CACHE_WH, "camera": c, "thresholds": {}}
        (pdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        made[name] = {"images": len(keep), "roi": roi, "boxes_used": len(boxes[c]),
                      "good": sum(1 for r in keep if not any(v == "1" for v in r[1:-1]))}
    return made


def selftest():
    b = [(0.4, 0.05, 0.6, 0.95)] * 50 + [(0.1, 0.0, 0.2, 0.2)]
    r = roi_from_boxes(b)
    assert r[0] <= int(0.4 * 3840) and r[0] + r[2] >= int(0.6 * 3840) and r[1] < 40
    print("camera_roi self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["build"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "build":
        print(json.dumps(build(), indent=1))
    else:
        ap.print_help()
