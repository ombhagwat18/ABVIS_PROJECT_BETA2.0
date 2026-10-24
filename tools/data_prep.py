"""Data preparation for the enclosure classification set (project enclosure_cls / enclosure_seg).

    python -m tools.data_prep hands      # candidates for hand / person in frame -> projects/enclosure_cls/prep/hands.json + sheet
    python -m tools.data_prep nolabel    # bottles with no label (teal label test) -> prep/nolabel.json + sheet
    python -m tools.data_prep camera     # the camera tag (A / B / unknown) per image -> prep/camera.json + sheet of the unknowns
    python -m tools.data_prep --selftest

Nothing here edits labels: every command writes a candidate list and a contact sheet for a person to confirm. The
confirmed changes are applied with dataset.delete_images / dataset.apply_labels (undoable, logged).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "NewDataset" / "metadata"
PREP = ROOT / "projects" / "enclosure_cls" / "prep"


def images():
    return [p for d in sorted(SRC.iterdir()) if d.is_dir() for p in sorted(d.glob("*.jpg"))]


def sheet(paths, out: Path, cols: int = 6, tw: int = 300, th: int = 169, labels=None):
    rows = max(1, -(-len(paths) // cols))
    img = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.resize(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4), (tw, th))
        y, x = (i // cols) * th, (i % cols) * tw
        img[y:y + th, x:x + tw] = im
        cv2.putText(img, (labels[i] if labels else p.stem[5:]), (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def camera_of(im_small: np.ndarray) -> str:
    """A = dark wall on the left edge, B = the warehouse scene on the left edge. A GUESS from the image edges."""
    g = cv2.cvtColor(im_small, cv2.COLOR_BGR2GRAY)
    w = g.shape[1]
    left, right = float(g[:, :w // 6].mean()), float(g[:, -w // 6:].mean())
    return "A" if left < right - 10 else "B" if right < left - 10 else "unknown"


def cmd_camera():
    out = {}
    for p in images():
        out[f"{p.parent.name}/{p.name}"] = camera_of(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8))
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "camera.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    unk = [p for p in images() if out[f"{p.parent.name}/{p.name}"] == "unknown"]
    if unk:
        sheet(unk[:48], PREP / "camera_unknown.jpg", labels=[f"{p.parent.name[:4]}{p.stem[5:]}" for p in unk[:48]])
    n = {k: list(out.values()).count(k) for k in ("A", "B", "unknown")}
    print(n, "unknown sheet:", PREP / "camera_unknown.jpg" if unk else "none")


def cmd_hands(conf: float = 0.12):
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "yolov8n.pt"))                     # COCO, already in the repo; person / hand-like objects
    cand = {}
    ps = images()
    for i in range(0, len(ps), 16):
        chunk = ps[i:i + 16]
        ims = [cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4) for p in chunk]
        for p, r in zip(chunk, m(ims, imgsz=640, conf=conf, verbose=False)):
            c = [(m.names[int(k)], float(s)) for k, s in zip(r.boxes.cls, r.boxes.conf) if m.names[int(k)] == "person"]
            if c:
                cand[f"{p.parent.name}/{p.name}"] = max(s for _, s in c)
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "hands.json").write_text(json.dumps(cand, indent=1), encoding="utf-8")
    top = sorted(cand, key=cand.get, reverse=True)[:60]
    sheet([SRC / k for k in top], PREP / "hands.jpg", labels=[f"{k.split('/')[0][:4]}{k[-8:-4]} {cand[k]:.2f}" for k in top])
    print(len(cand), "candidates; sheet", PREP / "hands.jpg")


def teal_present(im_small: np.ndarray, min_frac: float = 0.002) -> dict:
    """cap / label teal presence: label = a teal blob below the top 15% of the bottle band, cap = teal blob at the top."""
    from vision import autoseed
    cap, label = autoseed.teal_parts(im_small)
    return {"cap": cap is not None, "label": label is not None}


def cmd_nolabel():
    cand = {}
    for p in images():
        im = cv2.resize(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4), None, fx=0.5, fy=0.5)   # 480x270 like autoseed scale 0.125
        # autoseed.teal_parts thresholds assume scale 0.5 of 4K; rescale the frame to that size for the same pixel rules
        im = cv2.resize(im, (1920, 1080))
        t = teal_present(im)
        cand[f"{p.parent.name}/{p.name}"] = t
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "nolabel.json").write_text(json.dumps(cand, indent=0), encoding="utf-8")
    nl = [k for k, v in cand.items() if not v["label"]]
    per = {}
    for k in nl:
        per[k.split("/")[0]] = per.get(k.split("/")[0], 0) + 1
    sheet([SRC / k for k in nl[:60]], PREP / "nolabel.jpg", labels=[f"{k.split('/')[0][:4]}{k[-8:-4]}" for k in nl[:60]])
    print(len(nl), "without a teal label; per folder:", per)


HAND_CONF = 0.25
HAND_EXTRA = ("DamageBottle/snap_936.jpg", "WaterLevel/snap_1607.jpg", "TiltedCap/snap_1322.jpg")   # seen on the contact sheets
FOLDER_COL = {"DamageBottle": "damaged_bottle", "DamegeLable": "damaged_label", "MissingCap": "missing_cap",
              "SkewBottle": "skewed_bottle", "TiltedCap": "tilt_cap", "WaterLevel": "water_level"}


def cmd_apply():
    """Apply the confirmed prep results to project enclosure_cls: drop hand frames (trash, undoable), add missing_label to bottles
    with no label (a damaged_label image with NO label at all becomes missing_label only: a label that is absent is not damaged),
    and write the final camera tags."""
    from vision import dataset as D
    D.use_project("enclosure_cls")
    defects, labels = D.load_labels()
    by_name = {Path(r).name: r for r in labels}
    hands = json.loads((PREP / "hands.json").read_text())
    drop = sorted({k for k, v in hands.items() if v >= HAND_CONF} | set(HAND_EXTRA))
    drop_rel = [by_name[Path(k).name] for k in drop if Path(k).name in by_name]
    if drop_rel:
        print("hand frames removed:", D.delete_images(drop_rel) if hasattr(D, "delete_images") else None, len(drop_rel))
    (D.IMAGE_ROOT / "-ve" / "missing_label").mkdir(parents=True, exist_ok=True)      # a class folder = a new defect column
    D.load_labels()
    nl = json.loads((PREP / "nolabel.json").read_text())
    gone = {Path(k).name for k in drop}
    add, swap = [], []
    for k, v in nl.items():
        folder, name = k.split("/")
        if v["label"] or name in gone or name not in by_name:
            continue
        (swap if folder == "DamegeLable" else add).append(by_name[name])
    if add:
        D.apply_labels(add, "missing_label", 1)
    if swap:
        D.apply_labels(swap, "missing_label", 1)
        D.apply_labels(swap, "damaged_label", 0)
    cam = json.loads((PREP / "camera.json").read_text())
    cam = {k: ("B" if v == "unknown" else v) for k, v in cam.items()}        # the unknown ones (Good 388-440) are the camera-B view
    (PREP / "camera_final.json").write_text(json.dumps(cam, indent=0), encoding="utf-8")
    d2, l2 = D.load_labels()
    print("missing_label added:", len(add), "| damaged_label -> missing_label:", len(swap), "| columns:", d2,
          "| images now:", len(l2))


def selftest():
    a = np.full((90, 160, 3), 30, np.uint8)
    a[:, 120:] = 120
    assert camera_of(a) == "A"
    assert camera_of(a[:, ::-1].copy()) == "B"
    assert camera_of(np.full((90, 160, 3), 80, np.uint8)) == "unknown"
    print("data_prep self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["hands", "nolabel", "camera", "apply"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd:
        {"hands": cmd_hands, "nolabel": cmd_nolabel, "camera": cmd_camera, "apply": cmd_apply}[a.cmd]()
    else:
        ap.print_help()
