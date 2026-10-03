"""Finish and benchmark the remaining model work without touching live app state.

    python model_bench.py cls-test [STAMP ...]        # held-out test, once, per om_bottle checkpoint
    python model_bench.py cls-train --arch convnext_tiny --batch 16
    python model_bench.py yolo --model yolov8s.pt     # train/val/test/bench one extra candidate
    python model_bench.py yolo --model yolov8s.pt --resume   # finish an interrupted run of it
    python model_bench.py yolo-bench                  # re-time the deployed stage2_best.pt, same session
    python model_bench.py registry                    # -> models/MODEL_REGISTRY.json

What this file never does:
  * change projects/active.txt or projects/om_bottle/config.json (train.run()
    sets active_model + thresholds; both files are snapshotted and restored),
  * overwrite models/stage2_yolo/stage2_best.pt, training_metadata.json or
    test_eval/ (yolo_stage2_train.py owns those; candidates get their own names),
  * pick a model on test. Classification thresholds come from validation
    (metrics.json); test is scored once and written beside the checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import dataset as D

ROOT = Path(__file__).resolve().parent
CLS_PROJECT = "om_bottle"
MANIFEST = ROOT / "vision_dataset" / "manifests" / "unified_manifest.csv"
REGISTRY = ROOT / "models" / "MODEL_REGISTRY.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def isolated_project(name: str):
    """use_project(name) for this process only: restore active.txt and the
    project's config.json byte-for-byte afterwards, so the running app keeps
    its project, its active model and its thresholds."""
    active = D.ACTIVE_TXT.read_bytes() if D.ACTIVE_TXT.exists() else None
    cfg_path = D.PROJECTS / name / "config.json"
    cfg = cfg_path.read_bytes()
    try:
        D.use_project(name)
        yield
    finally:
        cfg_path.write_bytes(cfg)
        if active is not None:
            D.ACTIVE_TXT.write_bytes(active)
        D._cfg_cache = None


# ---------------------------------------------------------------- classification

def merged_scenes() -> dict[str, str]:
    """path -> scene id from vision_data.py's cross-folder merge (VISION_DATASET.md 2)."""
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, encoding="utf-8") as f:
        return {r["image_id"][4:]: r["scene_id"] for r in csv.DictReader(f)
                if r["source_dataset"] == "classification"}


def _score(probs, truths, defects, thr) -> dict:
    import train
    per, n = {}, len(probs)
    for i, d in enumerate(defects):
        pr, tr = probs[:, i], truths[:, i]
        if tr.sum() == 0:
            fp = int((pr >= thr[d]).sum())
            per[d] = {"n_pos": 0, "threshold": thr[d], "fp": fp, "note": "no test positives - unscored"}
            continue
        p, r, f, tp, fp, fn = train._pr(pr, tr, thr[d])
        per[d] = {"n_pos": int(tr.sum()), "threshold": thr[d], "precision": round(p, 4),
                  "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}
    scored = [x for x in per.values() if x["n_pos"]]
    T = np.array([thr[d] for d in defects], np.float32)
    pred = probs >= T
    truth = truths == 1
    verdict_ok = pred.any(1) == truth.any(1)
    return {"n_images": n,
            "macro_f1": round(float(np.mean([x["f1"] for x in scored])), 4),
            "macro_precision": round(float(np.mean([x["precision"] for x in scored])), 4),
            "macro_recall": round(float(np.mean([x["recall"] for x in scored])), 4),
            "exact_match_accuracy": round(float((pred == truth).all(1).mean()), 4),
            "label_accuracy": round(float((pred == truth).mean()), 4),
            "pass_reject_accuracy": round(float(verdict_ok.mean()), 4),
            "false_pass": int((~pred.any(1) & truth.any(1)).sum()),
            "false_reject": int((pred.any(1) & ~truth.any(1)).sum()),
            "per_defect": per}


def cls_test(stamp: str) -> dict:
    """Held-out test score at the VALIDATION-chosen thresholds, plus latency.

    train.evaluate_test() re-picks each threshold to maximise F1 on the set it
    scores; on test that is selection-on-test, so it is not used here.
    """
    import infer
    d = D.MODELS / stamp
    met = json.loads((d / "metrics.json").read_text())
    if not met.get("test_paths"):
        raise RuntimeError(f"{stamp} recorded no test set (trained before the held-out split) - not scorable")
    out_p = d / "test_metrics.json"
    if out_p.exists():
        print(f"{stamp}: test already scored -> {out_p.name} (not re-scored)")
        return json.loads(out_p.read_text())

    _, labels = D.load_labels()
    m = infer.Model(stamp)
    thr = {k: float(met["per_defect"][k]["threshold"]) for k in m.defects}
    paths = [p for p in met["test_paths"] if p in labels]
    views, T, used = [], [], []
    for p in paths:
        img = D.cached_crop(p, m.cfg)
        if img is None:
            continue
        views.append(D.center_crop(img, m.input_wh))
        T.append([float(labels[p].get(k, 0)) for k in m.defects])
        used.append(p)

    # latency: the real predict_view() path (normalise + forward + sigmoid), batch 1
    for v in views[:10]:
        m.predict_view(v)
    P, lat = [], []
    for v in views:
        if m.dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        pr = m.predict_view(v)
        lat.append((time.perf_counter() - t) * 1000)
        P.append([pr[k] for k in m.defects])
    probs, truths = np.array(P, np.float32), np.array(T, np.float32)

    m.net.to("cpu"); m.dev = "cpu"
    cpu = []
    for v in views[:5]:
        m.predict_view(v)
    for v in views[:30]:
        t = time.perf_counter(); m.predict_view(v); cpu.append((time.perf_counter() - t) * 1000)

    res = {"stamp": stamp, "arch": met["arch"], "scored_on": time.strftime("%Y-%m-%d %H:%M:%S"),
           "threshold_source": "validation (metrics.json per_defect.threshold)",
           "test": _score(probs, truths, m.defects, thr),
           "dropped": len(met["test_paths"]) - len(used)}

    # second view: drop test images whose cross-folder scene also appears in training
    sc = merged_scenes()
    if sc:
        held = set(met["test_paths"]) | set(met["val_paths"])
        train_scenes = {sc[p] for p in labels if p not in held and p in sc}
        keep = [i for i, p in enumerate(used) if sc.get(p) not in train_scenes]
        res["leak_excluded"] = len(used) - len(keep)
        res["test_leak_free"] = _score(probs[keep], truths[keep], m.defects, thr)
    a = np.array(lat)
    res["latency"] = {"device": "cuda" if torch.cuda.is_available() else "cpu",
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "batch": 1, "n": len(lat), "ms_mean": round(float(a.mean()), 2),
                      "ms_p50": round(float(np.percentile(a, 50)), 2),
                      "ms_p95": round(float(np.percentile(a, 95)), 2),
                      "fps": round(1000 / float(a.mean()), 1),
                      "cpu_ms_mean": round(float(np.mean(cpu)), 2),
                      "scope": "predict_view on a pre-cropped 192x448 view; excludes ROI crop/letterbox"}
    res["model_size_mb"] = round((d / "model.pt").stat().st_size / 2**20, 2)
    res["sha256"] = sha256(d / "model.pt")
    res["labels_csv_sha256"] = sha256(d / "labels.csv") if (d / "labels.csv").exists() else None
    out_p.write_text(json.dumps(res, indent=2))
    t = res["test"]
    print(f"{stamp} {met['arch']:<20} test macroF1={t['macro_f1']} P={t['macro_precision']} "
          f"R={t['macro_recall']} exact={t['exact_match_accuracy']} pass/reject={t['pass_reject_accuracy']} "
          f"leak-free F1={res.get('test_leak_free', {}).get('macro_f1')}  {res['latency']['ms_mean']} ms")
    return res


def cls_report(stamp: str) -> dict:
    """Per-class 2x2 confusion on the held-out test set, at the VALIDATION thresholds, including
    GOOD (= every defect below threshold; not a stored column). Re-scores the recorded test_paths
    with frozen thresholds: deterministic, no selection. Writes test_report.json."""
    import infer
    d = D.MODELS / stamp
    met = json.loads((d / "metrics.json").read_text())
    if not met.get("test_paths"):
        raise RuntimeError("no recorded test set")
    _, labels = D.load_labels()
    m = infer.Model(stamp)
    thr = {k: float(met["per_defect"][k]["threshold"]) for k in m.defects}
    P, T, used = [], [], []
    for p in met["test_paths"]:
        if p not in labels:
            continue
        img = D.cached_crop(p, m.cfg)
        if img is None:
            continue
        pr = m.predict_view(D.center_crop(img, m.input_wh))
        P.append([pr[k] for k in m.defects]); T.append([float(labels[p].get(k, 0)) for k in m.defects]); used.append(p)
    probs, truths = np.array(P, np.float32), np.array(T, np.float32) == 1
    pred = probs >= np.array([thr[k] for k in m.defects], np.float32)
    cols = {k: (pred[:, i], truths[:, i]) for i, k in enumerate(m.defects)}
    cols["GOOD"] = (~pred.any(1), ~truths.any(1))
    per = {}
    for k, (pp, tt) in cols.items():
        tp, fp = int((pp & tt).sum()), int((pp & ~tt).sum())
        fn, tn = int((~pp & tt).sum()), int((~pp & ~tt).sum())
        pr_ = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        per[k] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  "precision": None if pr_ is None else round(pr_, 4), "recall": None if rc is None else round(rc, 4),
                  "f1": round(2 * pr_ * rc / (pr_ + rc), 4) if pr_ and rc else (0.0 if tp + fn else None),
                  "threshold": thr.get(k), "fp_examples": [used[i] for i in np.where(pp & ~tt)[0][:5]],
                  "fn_examples": [used[i] for i in np.where(~pp & tt)[0][:5]]}
    rep = {"stamp": stamp, "arch": met["arch"], "n_test": len(used),
           "threshold_source": "validation (metrics.json)",
           "exact_match_accuracy": round(float((pred == truths).all(1).mean()), 4), "per_class": per}
    (d / "test_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def cmd_cls_report(a):
    with isolated_project(CLS_PROJECT):
        stamps = a.stamps or sorted(p.name for p in D.MODELS.iterdir()
                                    if (p / "metrics.json").exists() and json.loads((p / "metrics.json").read_text()).get("test_paths"))
        for s in stamps:
            r = cls_report(s)
            print(f"{s} {r['arch']:<20} exact={r['exact_match_accuracy']}  " + "  ".join(
                f"{k}:{v['tp']}/{v['support']} fp{v['fp']}" for k, v in r["per_class"].items() if v["support"] or v["fp"]))


def cmd_cls_test(a):
    with isolated_project(CLS_PROJECT):
        stamps = a.stamps or sorted(p.name for p in D.MODELS.iterdir() if (p / "metrics.json").exists())
        for s in stamps:
            try:
                cls_test(s)
            except RuntimeError as e:
                print(f"{s}: skipped - {e}")


def cmd_cls_train(a):
    import train
    with isolated_project(CLS_PROJECT):
        before = (D.PROJECTS / CLS_PROJECT / "config.json").read_bytes()
        s = train.run(epochs=a.epochs, batch=a.batch, arch=a.arch, patience=a.patience)
        print("trained", s["stamp"], "val macroF1", s["macro_f1"])
        # train.run just made this the project's active model; undo before anything else reads it
        (D.PROJECTS / CLS_PROJECT / "config.json").write_bytes(before)
        D._cfg_cache = None
        torch.cuda.empty_cache()
        cls_test(s["stamp"])


# ---------------------------------------------------------------- detection

def cmd_yolo(a):
    import platform
    import ultralytics
    from ultralytics import YOLO
    import yolo_stage2_train as Y

    if a.data == "v2":
        # cap = cap only (stage2_dataset/annotations_v2.py); same split and images as v1
        Y.EXPORT = ROOT / "stage2_dataset" / "yolo_export_v2"
        Y.DATA_YAML = Y.EXPORT / "data.yaml"
    sfx = "" if a.data == "v1" else f"_{a.data}"
    stem = Path(a.model).stem
    name = f"bench_{stem}{sfx}"
    meta_p = Y.OUT / f"candidate_{stem}{sfx}.json"
    if meta_p.exists():
        raise SystemExit(f"{meta_p} exists - delete it to retrain")
    dev = 0 if torch.cuda.is_available() else "cpu"
    mem = Y.mem_status()
    print("[mem]", mem, flush=True)
    before = Y.tree_hash()
    t0 = time.time()
    last = Y.OUT / name / "weights" / "last.pt"
    if a.resume:
        # an interrupted run: continue from its own last.pt with its own saved args
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(data=str(Y.DATA_YAML), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                    device=dev, seed=Y.SEED, deterministic=True, workers=a.workers, project=str(Y.OUT),
                    name=name, exist_ok=True, plots=True, val=True, verbose=False)
    train_min = (time.time() - t0) / 60
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    m = YOLO(str(best))
    kw = dict(data=str(Y.DATA_YAML), imgsz=a.imgsz, batch=a.batch, device=dev, workers=a.workers,
              verbose=False, project=str(Y.OUT), exist_ok=True)
    val = m.val(split="val", plots=False, name=f"{name}_valeval", **kw)
    test = m.val(split="test", plots=True, name=f"{name}_testeval", **kw)
    bench = Y.benchmark(m, "test", a.imgsz, dev)
    bench_cpu = Y.benchmark(YOLO(str(best)), "test", a.imgsz, "cpu", warm=3)
    removed = Y.clean_caches()
    after = Y.tree_hash()
    rows = (run_dir / "results.csv").read_text().splitlines()
    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"), model=a.model, run_dir=str(run_dir.relative_to(ROOT)),
        checkpoint=str(best.relative_to(ROOT)), sha256=sha256(best), size_mb=round(best.stat().st_size / 1e6, 2),
        n_params=sum(p.numel() for p in m.model.parameters()), train_minutes=round(train_min, 1),
        epochs_run=len(rows) - 1, resumed=a.resume,
        train_args=dict(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience, seed=Y.SEED,
                        deterministic=True, workers=a.workers, selection="best.pt = best val fitness"),
        note=("Test is reported for this candidate as a benchmark; selection between candidates must use "
              "val. stage2_best.pt / training_metadata.json were not modified."),
        val=Y.metrics_of(val, m.names), test=Y.metrics_of(test, m.names),
        benchmark_gpu=bench, benchmark_cpu=bench_cpu,
        dataset=dict(version=a.data, yaml=str(Y.DATA_YAML.relative_to(ROOT)),
                     annotations_file="stage2_dataset/annotations.json" if a.data == "v1"
                     else f"stage2_dataset/annotations_{a.data}.json",
                     split_json_sha256=sha256(ROOT / "stage2_dataset/split.json"),
                     annotations_sha256=sha256(ROOT / ("stage2_dataset/annotations.json" if a.data == "v1"
                                                       else f"stage2_dataset/annotations_{a.data}.json")),
                     export_tree_sha256=before, export_unchanged=before == after),
        removed_ultralytics_cache_files=removed, mem_before=mem,
        env=dict(ultralytics=ultralytics.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 python=platform.python_version()))
    assert before == after, "yolo_export changed during run!"
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("checkpoint", "size_mb", "val", "test", "benchmark_gpu")}, indent=2))


def cmd_yolo_bench(a):
    """Re-time the deployed detector in this session so its latency sits beside new candidates."""
    from ultralytics import YOLO
    import yolo_stage2_train as Y
    best = Y.OUT / "stage2_best.pt"
    dev = 0 if torch.cuda.is_available() else "cpu"
    out = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), checkpoint=str(best.relative_to(ROOT)),
               sha256=sha256(best), benchmark_gpu=Y.benchmark(YOLO(str(best)), "test", 640, dev),
               benchmark_cpu=Y.benchmark(YOLO(str(best)), "test", 640, "cpu", warm=3))
    (Y.OUT / "rebench_yolov8n.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def det_defects(weights: str, ann: str = "v2", split: str = "test", device="cpu") -> dict:
    """Defect-level score of a detector through the PRODUCTION rule (decision.detection_findings):
    per image, the inspected bottle (nearest the station) -> missing_cap / missing_label /
    cap_misplaced / no bottle, against the same rule applied to the ground-truth boxes.

    Ground truth comes from annotations_<ann>.json (v1 = annotations.json); presence of a cap or
    label does not depend on box tightness, so v1 and v2 give the same truth for these defects.
    """
    import cv2
    import decision as DEC
    import detect
    import yolo_stage2_train as Y
    af = ROOT / "stage2_dataset" / ("annotations.json" if ann == "v1" else f"annotations_{ann}.json")
    a = json.loads(af.read_text())["images"]
    split_j = json.loads((ROOT / "stage2_dataset" / "split.json").read_text())
    files = sorted(f for sc, fs in split_j["splits"][split]["files_by_scene"].items() for f in fs)
    det = detect.YoloDetector(weights=weights, verify=False, device=device, warmup=False)
    rules = dict(DEC.RULES)
    names = ("missing_cap", "missing_label", "cap_misplaced", "no_bottle")
    cm = {n: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_examples": [], "fn_examples": []} for n in names}
    per_image, lat = [], []
    for f in files:
        img = cv2.imread(str(ROOT / "stage2_dataset" / "clean" / f))
        H, W = img.shape[:2]
        gt_dets = tuple(detect.Detection(["bottle", "cap", "label"].index(b["cls"]), b["cls"], 1.0,
                                         (b["x"] - b["w"] / 2) * W, (b["y"] - b["h"] / 2) * H,
                                         (b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H)
                        for b in a[f]["boxes"])
        gt_res = detect.DetectionResult("gt", None, None, gt_dets, (W, H), 0.0, "gt", 1.0)
        g, _ = DEC.detection_findings(gt_res, rules)
        r = det.detect(img)
        lat.append(r.infer_ms)
        p, _ = DEC.detection_findings(r, rules)
        gs = {"no_bottle"} if g is None else set(g)
        ps = {"no_bottle"} if p is None else set(p)
        for n in names:
            k = ("tp" if n in ps else "fn") if n in gs else ("fp" if n in ps else "tn")
            cm[n][k] += 1
            if k in ("fp", "fn") and len(cm[n][f"{k}_examples"]) < 15:
                cm[n][f"{k}_examples"].append(f)
        per_image.append({"image": f, "truth": sorted(gs) or ["ok"], "pred": sorted(ps) or ["ok"],
                          "cap_conf": [round(d.confidence, 3) for d in r.detections if d.class_name == "cap"]})
    for n, c in cm.items():
        c["precision"] = round(c["tp"] / (c["tp"] + c["fp"]), 4) if c["tp"] + c["fp"] else None
        c["recall"] = round(c["tp"] / (c["tp"] + c["fn"]), 4) if c["tp"] + c["fn"] else None
        c["support"] = c["tp"] + c["fn"]
    good = sum(1 for x in per_image if x["truth"] == ["ok"])
    good_ok = sum(1 for x in per_image if x["truth"] == ["ok"] and x["pred"] == ["ok"])
    return {"weights": weights, "truth_from": af.name, "split": split, "n_images": len(files), "rules": rules,
            "device": str(device), "latency_ms_mean": round(float(np.mean(lat)), 1),
            "good": {"support": good, "passed": good_ok, "pass_rate": round(good_ok / good, 4) if good else None},
            "defects": cm, "per_image": per_image}


def cmd_det_defects(a):
    r = det_defects(a.weights, a.ann, a.split, a.device)
    out = ROOT / "models" / "stage2_yolo" / f"defects_{a.tag}_{a.split}.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"{a.tag} [{a.split}, truth {r['truth_from']}] GOOD {r['good']['passed']}/{r['good']['support']}")
    for n, c in r["defects"].items():
        print(f"  {n:<14} support={c['support']:<3} TP={c['tp']:<3} FP={c['fp']:<3} FN={c['fn']:<3} "
              f"P={c['precision']} R={c['recall']}  FP eg {c['fp_examples'][:3]}  FN eg {c['fn_examples'][:3]}")
    print("  ->", out)


# ---------------------------------------------------------------- registry

def cmd_registry(a):
    reg = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "Built from files on disk by model_bench.py registry; nothing re-measured here.",
           "classification": [], "detection": [], "segmentation": []}
    with isolated_project(CLS_PROJECT):
        active = D.load_config().get("active_model")
        for d in sorted(p for p in D.MODELS.iterdir() if (p / "metrics.json").exists()):
            m = json.loads((d / "metrics.json").read_text())
            t = json.loads((d / "test_metrics.json").read_text()) if (d / "test_metrics.json").exists() else None
            reg["classification"].append(dict(
                name=f"cls_{m['arch']}_{d.name}", stamp=d.name, arch=m["arch"], task="multi-label classification",
                project=CLS_PROJECT, active_in_app=d.name == active,
                checkpoint=str((d / "model.pt").relative_to(ROOT)),
                sha256=t["sha256"] if t else sha256(d / "model.pt"),
                size_mb=round((d / "model.pt").stat().st_size / 2**20, 2),
                config=dict(epochs=m["epochs"], epochs_run=m.get("epochs_run", m["epochs"]), batch=m["batch"],
                            lr=m["lr"], input_wh=m["input_wh"], device=m["device"], patience=m.get("patience")),
                dataset_version=dict(labels_csv_sha256=t["labels_csv_sha256"] if t else
                                     (sha256(d / "labels.csv") if (d / "labels.csv").exists() else None),
                                     n_train=m["n_train"], n_val=m["n_val"], n_test=m.get("n_test")),
                val=dict(macro_f1=m["macro_f1"], macro_precision=m.get("macro_precision"),
                         macro_recall=m.get("macro_recall"), accuracy=m.get("accuracy")),
                test=t and {k: t["test"][k] for k in t["test"] if k != "per_defect"},
                test_leak_free=t and t.get("test_leak_free") and
                {k: t["test_leak_free"][k] for k in t["test_leak_free"] if k != "per_defect"},
                latency_ms_gpu=t["latency"]["ms_mean"] if t else None,
                latency_ms_cpu=t["latency"]["cpu_ms_mean"] if t else None,
                train_time_latency_ms=m.get("infer_ms"),
                status="test-evaluated" if t else "legacy split (no held-out test recorded)"))
    import yolo_stage2_train as Y
    tm = json.loads((Y.OUT / "training_metadata.json").read_text())
    pv = json.loads((Y.OUT / "MODEL_PROVENANCE.json").read_text())
    reb = Y.OUT / "rebench_yolov8n.json"
    reb = json.loads(reb.read_text()) if reb.exists() else None
    reg["detection"].append(dict(
        name="det_yolov8n_stage2", arch="yolov8n", task="detection (bottle/cap/label)", deployed=True,
        checkpoint="models/stage2_yolo/stage2_best.pt", sha256=pv["model"]["checkpoint"]["sha256"],
        size_mb=round(tm["model_size_mb"], 2), n_params=tm["n_params"], trained=tm["date"],
        dataset_version=dict(export_tree_sha256=tm["export_hash_before"],
                             annotations_sha256=pv["dataset"]["annotation_schema"]["sha256"],
                             split_json_sha256=pv["dataset"]["split_file"]["sha256"]),
        val=tm["candidates"]["yolov8n.pt"]["val"], test=tm["test"],
        benchmark_gpu=(reb or tm)["benchmark_gpu"], benchmark_cpu=(reb or tm)["benchmark_cpu"]))
    for p in sorted(Y.OUT.glob("candidate_*.json")):
        c = json.loads(p.read_text())
        reg["detection"].append(dict(
            name=f"det_{Path(c['model']).stem}_stage2", arch=Path(c["model"]).stem,
            task="detection (bottle/cap/label)", deployed=False, checkpoint=c["checkpoint"], sha256=c["sha256"],
            size_mb=c["size_mb"], n_params=c["n_params"], trained=c["date"], dataset_version=c["dataset"],
            val=c["val"], test=c["test"], benchmark_gpu=c["benchmark_gpu"], benchmark_cpu=c["benchmark_cpu"]))
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print("wrote", REGISTRY)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cls-test"); s.add_argument("stamps", nargs="*")
    s = sub.add_parser("cls-report"); s.add_argument("stamps", nargs="*")
    s = sub.add_parser("cls-train")
    s.add_argument("--arch", required=True); s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--batch", type=int, default=32); s.add_argument("--patience", type=int)
    s = sub.add_parser("yolo")
    s.add_argument("--model", required=True); s.add_argument("--epochs", type=int, default=80)
    s.add_argument("--imgsz", type=int, default=640); s.add_argument("--batch", type=int, default=8)
    s.add_argument("--patience", type=int, default=20); s.add_argument("--workers", type=int, default=0)
    s.add_argument("--resume", action="store_true", help="continue an interrupted bench_<model> run")
    s.add_argument("--data", choices=["v1", "v2"], default="v1", help="v2 = cap-only annotations (annotations_v2.py)")
    sub.add_parser("yolo-bench")
    s = sub.add_parser("det-defects")
    s.add_argument("--weights", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--ann", default="v2"); s.add_argument("--split", default="test"); s.add_argument("--device", default="cpu")
    sub.add_parser("registry")
    a = ap.parse_args()
    {"cls-test": cmd_cls_test, "cls-report": cmd_cls_report, "cls-train": cmd_cls_train, "yolo": cmd_yolo,
     "yolo-bench": cmd_yolo_bench, "det-defects": cmd_det_defects, "registry": cmd_registry}[a.cmd](a)


if __name__ == "__main__":
    main()
