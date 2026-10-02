"""YOLO foundation / smoke test -- Stage 2D-A.

This module proves the annotation/export pipeline (annotate.py) can feed an
Ultralytics YOLO detection training run. It is deliberately isolated:

- It never touches projects/om_bottle (a classification project; irrelevant
  to YOLO anyway) or projects/bottle_detection (the real, hand-annotated
  detection project) -- the smoke test builds and trains on a tiny
  synthetic dataset in a temporary directory instead.
- It does not integrate with gui.py, train.py, or infer.py. Those are
  untouched.
- It does not build a model registry or any production inference path --
  see run_smoke_test()'s docstring for exactly what it does and does not
  prove.

Two independent pieces live here:

1. prepare_yolo_dataset() -- reusable, dataset-agnostic. Turns an
   annotate.py-shaped `data` dict plus the project's real image folder into
   the standard Ultralytics folder layout (images/{train,val,test},
   labels/{train,val,test}, data.yaml). It calls into annotate.py for
   validation and label-file generation rather than re-implementing the
   YOLO label format itself.

2. run_smoke_test() -- builds a tiny synthetic annotated "project" (a
   handful of generated images with one hand-placed box each), runs it
   through prepare_yolo_dataset(), then drives an actual tiny YOLO training
   + validation + inference cycle to prove the pipeline end-to-end.
"""
from __future__ import annotations

import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import annotate as A

# Where a smoke-test's YOLO weights are cached, deliberately OUTSIDE this
# project's own directory. Ultralytics downloads a bare model name (e.g.
# "yolov8n.pt") to the current working directory, which would otherwise
# leave a stray 6 MB file sitting in the project root on every fresh
# machine -- resolving an absolute path here keeps that out of the repo.
WEIGHTS_CACHE = Path.home() / ".cache" / "ultralytics_smoketest"
SMOKE_MODEL = "yolov8n.pt"          # smallest Ultralytics detection model


# ------------------------------------------------------------- dataset prep

def prepare_yolo_dataset(data: dict, image_root: Path, out_dir: Path,
                          val_frac: float = 0.2, test_frac: float = 0.15,
                          seed: int = 0) -> dict:
    """Build a standard Ultralytics YOLO detection dataset layout under
    `out_dir` from an annotate.py-shaped `data` dict.

        out_dir/
          images/{train,val,test}/
          labels/{train,val,test}/
          data.yaml

    `image_root` is the project's real images/ folder -- the same directory
    dataset.py's D.IMAGE_ROOT points at for the project this `data` belongs
    to. Every annotation's `boxes` and class references are validated (and
    every label line generated) by annotate.py, not re-implemented here.

    Behaviour, matching the Stage 2D-A requirements:
    - schema/coordinate/class validity: delegated entirely to
      annotate.validate_annotations() / annotate.export_yolo_detection(),
      called below -- an invalid box, an unknown class, or an out-of-range
      coordinate raises annotate.AnnotationError and aborts before any file
      is written.
    - missing image files: an annotation entry whose image does not exist
      under `image_root` is excluded from the split (reported back, not
      silently dropped -- see the returned "skipped_missing_image" list),
      rather than crashing the whole preparation over one absent file.
    - filename collisions: two different source images that would produce
      the same YOLO label stem (e.g. "a/x.jpg" and "b/x.jpg") are rejected
      outright, since silently letting one overwrite the other's label file
      would corrupt the dataset without any visible error.
    - split leakage: every image is placed in exactly one of
      train/val/test by construction (a single partition of one shuffled
      list), asserted before returning.
    - determinism: the split is shuffled with a fixed `seed`, so the same
      `data`/`image_root` always produces the same split.
    """
    if data.get("task") != "detection":
        raise ValueError(
            f"prepare_yolo_dataset only supports detection projects, got task={data.get('task')!r}")
    A.validate_annotations(data)             # raises AnnotationError on any schema violation

    image_root = Path(image_root)
    out_dir = Path(out_dir)

    candidates = sorted(data["images"])
    present, missing = [], []
    for rel in candidates:
        (present if (image_root / rel).is_file() else missing).append(rel)
    if not present:
        raise ValueError("no annotated image has a matching file under image_root -- nothing to prepare")

    stems = [Path(r).stem for r in present]
    dupes = [s for s, n in Counter(stems).items() if n > 1]
    if dupes:
        raise ValueError(
            f"filename collision(s) would corrupt the YOLO label mapping -- "
            f"more than one source image shares the stem(s) {dupes}; rename before exporting")

    order = list(present)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:]
    if not train:                             # a tiny dataset must not end up with nothing to train on
        train = [order[-1]]
        val = [p for p in val if p not in train]
        test = [p for p in test if p not in train]
    assert not (set(train) & set(val)), "train/val leakage"
    assert not (set(train) & set(test)), "train/test leakage"
    assert not (set(val) & set(test)), "val/test leakage"

    scratch = out_dir / "_scratch_labels"
    written = set(A.export_yolo_detection(data, scratch))   # labels only -- annotate.py's own format

    splits = {"train": train, "val": val, "test": test}
    for split, rels in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rel in rels:
            stem = Path(rel).stem
            shutil.copy2(image_root / rel, out_dir / "images" / split / f"{stem}{Path(rel).suffix}")
            src_lbl = scratch / f"{stem}.txt"
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            if rel in written and src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("")        # a background image (no boxes) is a valid YOLO example
    shutil.rmtree(scratch, ignore_errors=True)

    classes = list(data["classes"])           # ordered -- index is the YOLO class id, never hard-coded
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n", encoding="utf-8")

    return {"train": len(train), "val": len(val), "test": len(test), "total": len(present),
            "skipped_missing_image": missing, "data_yaml": str(out_dir / "data.yaml")}


# --------------------------------------------------------------- smoke test

def _synthetic_project(root: Path, n_images: int = 8):
    """A tiny made-up detection project: n_images small solid-background
    images, each with one filled rectangle ("widget") at a random-but-known
    location, and an annotate.py structure with exactly matching boxes.
    Entirely synthetic -- not derived from, and never written into,
    projects/bottle_detection or projects/om_bottle."""
    import cv2
    import numpy as np

    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    data = A.init_annotations("detection", ["widget"])

    rng = random.Random(0)
    size = 64
    for i in range(n_images):
        img = np.full((size, size, 3), 40, np.uint8)
        w, h = rng.randint(16, 28), rng.randint(16, 28)
        x0, y0 = rng.randint(0, size - w), rng.randint(0, size - h)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 200, 60), -1)
        rel = f"img{i:03d}.jpg"
        cv2.imwrite(str(image_root / rel), img)
        box = {"cls": "widget", "x": (x0 + w / 2) / size, "y": (y0 + h / 2) / size,
               "w": w / size, "h": h / size}
        A.set_image_annotation(data, rel, boxes=[box], reviewed=True)

    A.validate_annotations(data)
    return data, image_root


def _resolve_weights() -> Path:
    """The smallest Ultralytics detection checkpoint, cached outside this
    project's directory. Downloads on first use (requires network); reuses
    the cached copy on every run after that."""
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_CACHE / SMOKE_MODEL
    from ultralytics import YOLO
    YOLO(str(target))          # downloads to `target` itself if missing; no-op if already cached
    return target


def run_smoke_test(workdir: Path | None = None) -> dict:
    """End-to-end pipeline proof, entirely inside a temporary directory:

        synthetic annotations -> prepare_yolo_dataset() -> YOLO(...).train()
        -> YOLO(...).val() -> YOLO(...).predict()

    This is an ENGINEERING smoke test, not a model-quality test: 1 epoch,
    64x64 images, 8 total images. It proves the pipeline wires together and
    produces the expected artifacts; it makes no claim about detection
    accuracy, and nothing here is a production inference path.
    """
    import tempfile

    import torch
    from ultralytics import YOLO

    own_tmp = workdir is None
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="yolo_smoketest_"))
    report: dict = {"workdir": str(workdir)}
    try:
        data, image_root = _synthetic_project(workdir / "source")
        prep = prepare_yolo_dataset(data, image_root, workdir / "yolo_ds",
                                     val_frac=0.25, test_frac=0.125, seed=0)
        report["dataset_prep"] = prep
        assert prep["train"] > 0 and prep["val"] > 0, "smoke dataset produced an empty split"

        weights = _resolve_weights()
        report["weights"] = str(weights)
        device = 0 if torch.cuda.is_available() else "cpu"
        report["device"] = "cuda:0" if device == 0 else "cpu"

        model = YOLO(str(weights))
        # amp=False: Ultralytics' AMP self-check otherwise does its own bare
        # YOLO("yolov8n.pt") load, independent of the cached `weights` path
        # above, which downloads a second copy into the current working
        # directory (i.e. this project's root) on a machine with no cached
        # copy. Skipping that check is fine for a CPU/GPU pipeline smoke
        # test that isn't measuring training speed or numerical precision.
        results = model.train(data=prep["data_yaml"], epochs=1, imgsz=64, batch=2,
                               device=device, project=str(workdir / "runs"), name="smoke",
                               exist_ok=True, verbose=False, plots=False, workers=0, amp=False)
        run_dir = Path(results.save_dir)
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        ckpt = best if best.exists() else last
        assert ckpt.exists(), f"no checkpoint produced under {run_dir}"
        report["training"] = {"run_dir": str(run_dir), "checkpoint": str(ckpt)}

        trained = YOLO(str(ckpt))
        val_metrics = trained.val(data=prep["data_yaml"], imgsz=64, device=device,
                                   project=str(workdir / "runs"), name="smoke_val",
                                   exist_ok=True, plots=False, verbose=False)
        report["validation"] = {"ran": True, "keys": list(vars(val_metrics.box).keys())
                                 if hasattr(val_metrics, "box") else []}

        test_dir = workdir / "yolo_ds" / "images" / "test"
        probe_dir = test_dir if any(test_dir.iterdir()) else workdir / "yolo_ds" / "images" / "train"
        probe_img = next(probe_dir.iterdir())
        preds = trained.predict(source=str(probe_img), imgsz=64, device=device, verbose=False)
        assert len(preds) == 1 and hasattr(preds[0], "boxes"), "inference returned no detections structure"
        report["inference"] = {"image": str(probe_img), "n_results": len(preds),
                                "n_boxes": int(preds[0].boxes.shape[0]) if preds[0].boxes is not None else 0}

        report["ok"] = True
        return report
    finally:
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    try:
        r = run_smoke_test()
    except Exception as e:
        print(f"SMOKE TEST FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    print("ok  dataset prep:", r["dataset_prep"])
    print("ok  device:", r["device"])
    print("ok  training checkpoint:", r["training"]["checkpoint"])
    print("ok  validation ran:", r["validation"]["ran"])
    print("ok  inference on", r["inference"]["image"], "->", r["inference"]["n_boxes"], "box(es)")
    print("ok  yolo_train.py smoke test complete")
