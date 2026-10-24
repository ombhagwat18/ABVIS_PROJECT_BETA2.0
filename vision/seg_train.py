"""Segmentation model for the enclosure dataset (bottle / cap / label masks + defect regions).

    python -m vision.seg_train export  --project enclosure_seg            # reviewed images -> YOLO-seg dataset, split by burst
    python -m vision.seg_train train   --project enclosure_seg --model n  # train on train, pick on val (writes models/enclosure_seg/<run>)
    python -m vision.seg_train test    --run <run>                        # the held-out test split, scored ONCE, written to the run folder
    python -m vision.seg_train --selftest

Rules (same as the rest of the repo):
  * only images a person REVIEWED are exported (annotate.exportable); an unreviewed auto-seed never trains a model,
    unless --include-unreviewed is given, which is for a smoke test and is stamped "BOOTSTRAP, NOT FOR RESULTS";
  * the split is by BURST (30 consecutive frames of one folder = one block) and stratified per folder, so two
    near-identical frames never sit on both sides of the split;
  * ultralytics gets workers= on train() and val() (Windows paging-file lesson), nothing is downloaded;
  * the test split is evaluated once per run and the number is stored, never used to choose anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision import annotate as A                                     # noqa: E402

BLOCK = 30
SPLIT = (0.70, 0.15, 0.15)
MAX_W = 1280
WEIGHTS = {"n": ROOT / "models" / "pretrained" / "yolov8n-seg.pt",
           "s": ROOT / "models" / "pretrained" / "yolov8s-seg.pt"}


def pdir(slug: str) -> Path:
    return ROOT / "projects" / slug


def _num(rel: str) -> int:
    m = re.search(r"(\d+)", Path(rel).stem)
    return int(m.group(1)) if m else 0


def burst_split(rels, folder_of, seed: int = 0) -> dict:
    """{rel: 'train'|'val'|'test'}: blocks of BLOCK consecutive frames per folder, shuffled with a fixed seed,
    cut 70/15/15 per folder, so every class appears in every split and bursts never straddle a boundary."""
    out = {}
    by = {}
    for r in rels:
        by.setdefault(folder_of(r), []).append(r)
    for folder, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=_num)
        blocks = [rs[i:i + BLOCK] for i in range(0, len(rs), BLOCK)]
        rnd = random.Random(f"{seed}:{folder}")
        rnd.shuffle(blocks)
        n = len(blocks)
        n_val = max(1, round(n * SPLIT[1])) if n >= 3 else 0
        n_test = max(1, round(n * SPLIT[2])) if n >= 3 else 0
        for i, b in enumerate(blocks):
            name = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            for r in b:
                out[r] = name
    return out


def export(slug: str, include_unreviewed: bool = False, out: Path | None = None) -> dict:
    p = pdir(slug)
    data = A.load_annotations(p / "annotations.json")
    meta = json.loads((p / "image_meta.json").read_text(encoding="utf-8"))
    classes = data["classes"]
    ids = {c: i for i, c in enumerate(classes)}
    rels = [r for r, e in data["images"].items()
            if (A.exportable(e, "polygons") and e.get("reviewed")) or (include_unreviewed and e.get("polygons"))]
    if not rels:
        raise SystemExit("no reviewed images yet: review in the Annotation Studio first (or --include-unreviewed to smoke-test)")
    split = burst_split(rels, lambda r: meta.get(r, {}).get("folder"))
    out = out or (p / "yolo_seg")
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": 0, "val": 0, "test": 0}
    objs = {c: 0 for c in classes}
    for rel in rels:
        sp = split[rel]
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        im = cv2.imread(str(p / "images" / rel))
        if im.shape[1] > MAX_W:
            im = cv2.resize(im, (MAX_W, round(im.shape[0] * MAX_W / im.shape[1])), interpolation=cv2.INTER_AREA)
        stem = Path(rel).stem
        cv2.imwrite(str(out / "images" / sp / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for poly in data["images"][rel].get("polygons") or []:
            lines.append(f"{ids[poly['cls']]} " + " ".join(f"{v:.6f}" for v in poly["points"]))
            objs[poly["cls"]] += 1
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[sp] += 1
    yaml = (f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    ann_sha = hashlib.sha256((p / "annotations.json").read_bytes()).hexdigest()
    man = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "project": slug, "images": counts, "objects": objs,
           "classes": classes, "split": f"burst blocks of {BLOCK}, stratified per folder, seed 0",
           "annotations_sha256": ann_sha,
           "stamp": "BOOTSTRAP, NOT FOR RESULTS" if include_unreviewed else "reviewed images only"}
    (out / "export_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    (out / "split.json").write_text(json.dumps(split, indent=0), encoding="utf-8")
    return man


def train(slug: str, size: str = "n", epochs: int = 80, imgsz: int = 960, batch: int = 4, device: str = "0") -> Path:
    from ultralytics import YOLO
    p = pdir(slug)
    ex = p / "yolo_seg"
    man = json.loads((ex / "export_manifest.json").read_text(encoding="utf-8"))
    run = time.strftime("%Y%m%d-%H%M%S") + f"_{size}"
    proj = ROOT / "models" / "enclosure_seg"
    w = WEIGHTS.get(size) or ROOT / "models" / "pretrained" / f"yolov8{size}-seg.pt"
    if not w.exists():
        raise SystemExit(f"{w} missing (weights are never downloaded implicitly)")
    m = YOLO(str(w))
    m.train(data=str(ex / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, workers=2, device=device,
            project=str(proj), name=run, exist_ok=False, seed=0, patience=25, cos_lr=True, plots=True,
            fliplr=0.0, flipud=0.0, mosaic=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=3.0, translate=0.08,
            scale=0.35, verbose=False)
    run_dir = proj / run
    (run_dir / "export_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    (run_dir / "RUN.json").write_text(json.dumps({"project": slug, "size": size, "epochs": epochs, "imgsz": imgsz,
                                                    "batch": batch, "weights": str(w.name),
                                                    "weights_sha256": hashlib.sha256(w.read_bytes()).hexdigest(),
                                                    "data_stamp": man["stamp"]}, indent=1), encoding="utf-8")
    return run_dir


def test(run: str, device: str = "0") -> dict:
    from ultralytics import YOLO
    run_dir = ROOT / "models" / "enclosure_seg" / run
    done = run_dir / "TEST_RESULT.json"
    if done.exists():
        raise SystemExit(f"{done} exists: the test split is evaluated once per run")
    info = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
    ex = pdir(info["project"]) / "yolo_seg"
    m = YOLO(str(run_dir / "weights" / "best.pt"))
    r = m.val(data=str(ex / "data.yaml"), split="test", imgsz=info["imgsz"], batch=info["batch"], workers=2,
              device=device, verbose=False, project=str(run_dir), name="test", exist_ok=True)
    names = m.names
    res = {"run": run, "split": "test", "data_stamp": info["data_stamp"], "images": json.loads(
        (run_dir / "export_manifest.json").read_text(encoding="utf-8"))["images"],
        "box_map50": float(r.box.map50), "box_map50_95": float(r.box.map),
        "mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map),
        "per_class_mask_map50": {names[int(c)]: float(v) for c, v in zip(r.ap_class_index, r.seg.ap50)},
        "per_class_mask_map50_95": {names[int(c)]: float(v) for c, v in zip(r.ap_class_index, r.seg.ap)},
        "speed_ms": {k: float(v) for k, v in r.speed.items()}}
    done.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def selftest():
    rels = [f"_inbox/snap_{i:04d}.jpg" for i in range(1, 301)]
    folder = lambda r: "A" if _num(r) <= 150 else "B"                # noqa: E731
    sp = burst_split(rels, folder)
    assert set(sp.values()) == {"train", "val", "test"}
    assert sp == burst_split(rels, folder), "deterministic"
    for f in "AB":
        for name in ("train", "val", "test"):
            assert any(sp[r] == name for r in rels if folder(r) == f), (f, name)
    blocks = {}
    for r in rels:
        blocks.setdefault((folder(r), (sorted(x for x in rels if folder(x) == folder(r)).index(r)) // BLOCK), set()).add(sp[r])
    assert all(len(v) == 1 for v in blocks.values()), "a burst straddles two splits"
    print("seg_train self-test ok (burst split: deterministic, stratified, bursts intact)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["export", "train", "test"])
    ap.add_argument("--project", default="enclosure_seg")
    ap.add_argument("--model", default="n")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--run")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "export":
        print(json.dumps(export(a.project, a.include_unreviewed), indent=1))
    elif a.cmd == "train":
        print(train(a.project, a.model, a.epochs, a.imgsz, a.batch))
    elif a.cmd == "test":
        print(json.dumps(test(a.run), indent=1))
    else:
        ap.print_help()
