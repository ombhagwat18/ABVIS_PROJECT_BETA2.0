"""Stage 2 YOLO detector: train on train, select on val, evaluate test once.

Read-only against stage2_dataset/ (annotations, split.json, yolo_export). All
outputs go to models/stage2_yolo/. Ultralytics drops labels.cache files next to
the label folders; those are removed afterwards and the export is hashed
before/after to prove nothing else changed.

    python yolo_stage2_train.py                 # train candidates, pick by val, test once
    python yolo_stage2_train.py --candidates yolov8n.pt --epochs 2   # quick check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "stage2_dataset" / "yolo_export"
DATA_YAML = EXPORT / "data.yaml"
OUT = ROOT / "models" / "stage2_yolo"
WEIGHTS_CACHE = Path.home() / ".cache" / "ultralytics_smoketest"
SEED = 0


def mem_status() -> dict:
    """Free physical RAM and free commit (RAM+pagefile) in MB, via Win32 GlobalMemoryStatusEx."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                    ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(free_ram_mb=int(m.ap / 2**20), free_commit_mb=int(m.apf / 2**20), total_ram_mb=int(m.tp / 2**20))


def tree_hash() -> str:
    """Digest of every file under the export except Ultralytics' *.cache files."""
    h = hashlib.sha256()
    for p in sorted(EXPORT.rglob("*")):
        if p.is_file() and p.suffix != ".cache":
            h.update(str(p.relative_to(EXPORT)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def clean_caches() -> list[str]:
    gone = []
    for p in EXPORT.rglob("*.cache"):
        p.unlink()
        gone.append(str(p.relative_to(EXPORT)))
    return gone


def metrics_of(m, names) -> dict:
    b = m.box
    per = {}
    for k, ci in enumerate(b.ap_class_index):
        per[names[int(ci)]] = dict(precision=float(b.p[k]), recall=float(b.r[k]),
                                   map50=float(b.ap50[k]), map50_95=float(b.ap[k]))
    return dict(map50=float(b.map50), map50_95=float(b.map), precision=float(b.mp),
                recall=float(b.mr), per_class=per)


def benchmark(model, split: str, imgsz: int, device, warm=10) -> dict:
    """End-to-end predict() latency (preprocess+inference+NMS) on real test images, batch 1."""
    files = sorted((EXPORT / "images" / split).glob("*.png"))
    imgs = [cv2.imread(str(f)) for f in files]          # decode outside the timed region
    kw = dict(imgsz=imgsz, device=device, verbose=False, conf=0.001)
    for im in imgs[:warm]:
        model.predict(im, **kw)
    if device != "cpu":
        torch.cuda.synchronize()
    lat, inf = [], []
    for im in imgs:
        t = time.perf_counter()
        r = model.predict(im, **kw)
        if device != "cpu":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
        inf.append(r[0].speed["inference"])
    a = np.array(lat)
    return dict(n_images=len(imgs), device=str(device), batch=1,
                latency_ms_mean=float(a.mean()), latency_ms_p50=float(np.percentile(a, 50)),
                latency_ms_p95=float(np.percentile(a, 95)), fps_end_to_end=float(1000 / a.mean()),
                model_inference_ms_mean=float(np.mean(inf)), fps_model_only=float(1000 / np.mean(inf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="0 = in-process loading (robust on low-RAM Windows)")
    ap.add_argument("--min-free-ram-mb", type=int, default=3000,
                    help="do not start a further candidate below this much free RAM")
    ap.add_argument("--min-free-commit-mb", type=int, default=4000,
                    help="...or below this much free commit (RAM+pagefile); workers die at the commit limit")
    ap.add_argument("--name", default="run")
    a = ap.parse_args()

    from ultralytics import YOLO
    import ultralytics

    dev = 0 if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    before = tree_hash()
    print("export hash before:", before)

    results, skipped = {}, {}
    for i, cand in enumerate(a.candidates):
        mem = mem_status()
        print(f"[mem] before {cand}: {mem}", flush=True)
        if i > 0 and (mem["free_ram_mb"] < a.min_free_ram_mb or mem["free_commit_mb"] < a.min_free_commit_mb):
            skipped[cand] = dict(reason="insufficient memory before launch", mem=mem,
                                 min_free_ram_mb=a.min_free_ram_mb, min_free_commit_mb=a.min_free_commit_mb)
            print(f"[mem] SKIPPING {cand}: {skipped[cand]}", flush=True)
            continue
        wpath = WEIGHTS_CACHE / cand
        model = YOLO(str(wpath) if wpath.exists() else cand)
        t0 = time.time()
        model.train(data=str(DATA_YAML), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                    patience=a.patience, device=dev, seed=SEED, deterministic=True, workers=a.workers,
                    project=str(OUT), name=f"{a.name}_{Path(cand).stem}", exist_ok=True,
                    plots=True, val=True, verbose=False)
        train_min = (time.time() - t0) / 60
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"         # best.pt is chosen by val fitness
        m = YOLO(str(best))
        val = m.val(data=str(DATA_YAML), split="val", imgsz=a.imgsz, batch=a.batch, device=dev, workers=a.workers,
                    plots=False, verbose=False, project=str(OUT), name=f"{run_dir.name}_valeval",
                    exist_ok=True)
        results[cand] = dict(run_dir=str(run_dir), best=str(best), train_minutes=train_min,
                             epochs_run=len((run_dir / "results.csv").read_text().splitlines()) - 1,
                             val=metrics_of(val, m.names))
        print(cand, "val mAP50-95 =", results[cand]["val"]["map50_95"])

    # model selection uses VAL ONLY
    chosen = max(results, key=lambda c: results[c]["val"]["map50_95"])
    best = Path(results[chosen]["best"])
    m = YOLO(str(best))
    # held-out test: evaluated exactly once, for the chosen model only
    test = m.val(data=str(DATA_YAML), split="test", imgsz=a.imgsz, batch=a.batch, device=dev, workers=a.workers,
                 plots=True, verbose=False, project=str(OUT), name="test_eval", exist_ok=True)
    test_m = metrics_of(test, m.names)
    bench = benchmark(m, "test", a.imgsz, dev)
    bench_cpu = benchmark(YOLO(str(best)), "test", a.imgsz, "cpu", warm=3)

    final = OUT / "stage2_best.pt"
    final.write_bytes(best.read_bytes())

    removed = clean_caches()
    after = tree_hash()
    assert before == after, "yolo_export changed during run!"

    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"), chosen=chosen, model_path=str(final),
        model_size_mb=final.stat().st_size / 1e6,
        n_params=sum(p.numel() for p in m.model.parameters()),
        train_args=dict(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                        seed=SEED, deterministic=True, workers=a.workers, candidates=a.candidates,
                        selection="best val mAP50-95 (fitness) per run; candidate chosen on val mAP50-95"),
        data=dict(yaml=str(DATA_YAML), split_json="stage2_dataset/split.json",
                  train=418, val=89, test=87, classes=list(m.names.values())),
        candidates=results, candidates_skipped=skipped, test=test_m, benchmark_gpu=bench, benchmark_cpu=bench_cpu,
        export_hash_before=before, export_hash_after=after, export_unchanged=before == after,
        removed_ultralytics_cache_files=removed,
        env=dict(ultralytics=ultralytics.__version__, torch=torch.__version__,
                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 python=platform.python_version()),
    )
    (OUT / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("chosen", "model_size_mb", "test", "benchmark_gpu", "benchmark_cpu", "export_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
