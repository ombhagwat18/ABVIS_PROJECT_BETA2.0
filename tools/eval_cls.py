"""Accuracy AND confidence of one classifier checkpoint on its held-out test images, scored once.

    python -m tools.eval_cls enclosure_cama <stamp>
    python -m tools.eval_cls --selftest

Writes TEST_ANALYSIS.json next to the checkpoint (refuses to run twice: the test set is not a tuning set). Reports
  - per defect: n, precision, recall, F1 at the checkpoint's thresholds
  - bottle level: false-reject rate (good bottle flagged), miss rate (defective bottle passed), UNSURE share
  - confidence: mean score on the true positives / on the false alarms, a 10-bin reliability table (stated vs observed rate)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import dataset as D                                           # noqa: E402

PASS_MAX = 0.10                               # the production three-zone rule: between pass_max and the threshold = UNSURE


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list:
    """[(lo, hi, n, mean_score, observed_rate)] over every (image, defect) pair."""
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        out.append((round(float(lo), 2), round(float(hi), 2), int(m.sum()),
                    round(float(p[m].mean()), 3) if m.any() else None, round(float(y[m].mean()), 3) if m.any() else None))
    return out


def analyse(P: np.ndarray, T: np.ndarray, defects: list, thr: dict) -> dict:
    th = np.array([float(thr.get(d, 0.5)) for d in defects])
    flag = P >= th
    per = {}
    for j, d in enumerate(defects):
        tp = int((flag[:, j] & (T[:, j] > 0)).sum())
        fp = int((flag[:, j] & (T[:, j] == 0)).sum())
        fn = int((~flag[:, j] & (T[:, j] > 0)).sum())
        pr = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        f1 = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if tp + fn else None)
        per[d] = {"n_pos": int((T[:, j] > 0).sum()), "threshold": round(float(th[j]), 3), "tp": tp, "fp": fp, "fn": fn,
                  "precision": None if pr is None else round(pr, 3), "recall": None if rc is None else round(rc, 3),
                  "f1": None if f1 is None else round(f1, 3)}
    good = ~(T > 0).any(axis=1)
    bad = ~good
    unsure = ((P >= PASS_MAX) & ~flag).any(axis=1)
    tp_scores = P[flag & (T > 0)]
    fp_scores = P[flag & (T == 0)]
    return {"n_images": int(len(P)), "n_good": int(good.sum()), "n_defective": int(bad.sum()),
            "per_defect": per,
            "false_reject_rate": round(float(flag[good].any(axis=1).mean()), 3) if good.any() else None,
            "miss_rate": round(float((~flag[bad].any(axis=1)).mean()), 3) if bad.any() else None,
            "unsure_share_of_unflagged": round(float(unsure[~flag.any(axis=1)].mean()), 3) if (~flag.any(axis=1)).any() else None,
            "conf_mean_true_positive": round(float(tp_scores.mean()), 3) if tp_scores.size else None,
            "conf_mean_false_alarm": round(float(fp_scores.mean()), 3) if fp_scores.size else None,
            "reliability": reliability(P.ravel(), (T > 0).ravel().astype(float))}


def run(project: str, stamp: str) -> dict:
    from vision import infer
    D.use_project(project)
    out_path = D.MODELS / stamp / "TEST_ANALYSIS.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: the test set is scored once per checkpoint")
    mp = json.loads((D.MODELS / stamp / "metrics.json").read_text())
    defects, labels = D.load_labels()
    m = infer.Model(stamp)
    thr = {d: v["threshold"] for d, v in mp["per_defect"].items()}          # chosen on validation by the trainer
    paths = [p for p in mp["test_paths"] if p in labels]
    P, T = [], []
    for p in paths:
        img = D.cached_crop(p, m.cfg)
        if img is None:
            continue
        pr = m.predict_view(D.center_crop(img, m.input_wh))
        P.append([pr[d] for d in m.defects])
        T.append([float(labels[p].get(d, 0)) for d in m.defects])
    res = analyse(np.array(P, np.float32), np.array(T, np.float32), list(m.defects), thr)
    res.update({"project": project, "stamp": stamp, "dropped": len(mp["test_paths"]) - len(P)})
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def selftest():
    P = np.array([[0.9, 0.0], [0.1, 0.8], [0.05, 0.02], [0.4, 0.3]], np.float32)
    T = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], np.float32)
    r = analyse(P, T, ["a", "b"], {"a": 0.5, "b": 0.5})
    assert r["per_defect"]["a"]["recall"] == 1.0 and r["per_defect"]["b"]["precision"] == 1.0
    assert r["false_reject_rate"] == 0.0 and r["miss_rate"] == 0.0 and r["n_good"] == 2
    assert sum(b[2] for b in r["reliability"]) == 8
    print("eval_cls self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?")
    ap.add_argument("stamp", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.project and a.stamp:
        print(json.dumps(run(a.project, a.stamp), indent=1))
    else:
        ap.print_help()
