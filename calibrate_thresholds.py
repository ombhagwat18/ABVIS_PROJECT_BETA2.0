"""Pick per-defect decision thresholds from VALIDATION scores, check them on the held-out TEST set, apply on request.

Why: the live thresholds in config.json were hand-tuned low (0.05 - 0.2 for most defects), because validation scores
are saturated and any value separates them. On real camera frames a low threshold turns ordinary score noise into
"defect": a good bottle is flagged. This tool chooses each threshold where it matters:

  * grid 0.05 ... 0.95; per defect, the threshold with the best validation F1, ties -> the HIGHER threshold;
  * never below `floor` (default 0.30) unless that would lose validation recall below `min_recall` (0.95);
  * defects without validation positives are left as they are (0.5 / the disabled 1.01), never invented.

It reports, before and after, on VAL and on the TEST set the model never saw: per-defect false positives / misses and
the image-level numbers that matter on a line: GOOD bottles wrongly called defective, defective bottles passed.
Test is only READ here: it is never used to choose a threshold.

    python calibrate_thresholds.py [STAMP]              # report only (STAMP = active model by default)
    python calibrate_thresholds.py [STAMP] --apply      # write thresholds to config.json (the old ones are kept in
                                                        # config.json "thresholds_before_calibration")
    python calibrate_thresholds.py --restore            # undo the last --apply
    python calibrate_thresholds.py --selftest
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import dataset as D

GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def pick(prob: np.ndarray, truth: np.ndarray, floor=0.30, min_recall=0.95) -> tuple:
    """(threshold, note). prob/truth 1-D over the validation images of one defect."""
    npos = int(truth.sum())
    if npos == 0:
        return None, "no validation positives: left as it is"
    rows = []
    for t in GRID:
        pred = prob >= t
        tp = int((pred & (truth == 1)).sum())
        fp = int((pred & (truth == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / npos
        rows.append((t, 2 * p * r / (p + r) if p + r else 0.0, r))
    best_f = max(f for _, f, _ in rows)
    ok = [(t, f, r) for t, f, r in rows if r >= min_recall]
    if not ok:
        return max(rows, key=lambda x: (x[1], x[0]))[0], f"no threshold keeps recall >= {min_recall:g}: best F1 {best_f:.3f}"
    top = max(f for _, f, _ in ok)
    plateau = [t for t, f, _ in ok if f >= top - 1e-9]
    # validation is often saturated (every threshold in a wide band is perfect): take the MIDDLE of that band,
    # the point farthest from both the noisy negatives and the weakest positives
    t = plateau[len(plateau) // 2]
    if t < floor:
        higher = [x for x in ok if x[0] >= floor]
        if higher:
            t = max(higher, key=lambda x: (x[1], -abs(x[0] - 0.5)))[0]
    return float(t), f"val F1 {top:.3f}, perfect band {plateau[0]:.2f}-{plateau[-1]:.2f}" if len(plateau) > 1 else f"val F1 {top:.3f}"


def image_level(prob: np.ndarray, truth: np.ndarray, thr: np.ndarray) -> dict:
    """prob/truth (n, d); thr (d,). GOOD = no defect flag in truth."""
    pred = prob >= thr
    good = truth.sum(axis=1) == 0
    bad = ~good
    return {"n": int(len(prob)), "good": int(good.sum()), "defective": int(bad.sum()),
            "good_called_defective": int((pred.any(axis=1) & good).sum()),
            "defective_passed": int((~pred.any(axis=1) & bad).sum()),
            "per_defect_fp": {}, }


def _scores(stamp: str, key: str) -> tuple:
    """(defects, probs (n,d), truth (n,d), paths) of the checkpoint's own recorded val/test paths."""
    import infer
    metrics = json.loads((D.MODELS / stamp / "metrics.json").read_text())
    wanted = metrics.get(key) or []
    m = infer.Model(stamp)
    defects, all_labels = D.load_labels()
    labels = {p: r for p, r in all_labels.items() if r.get("reviewed", 1)}
    shared = [d for d in m.defects if d in defects]
    P, T, used = [], [], []
    for p in wanted:
        if p not in labels:
            continue
        img = D.cached_crop(p, m.cfg)
        if img is None:
            continue
        pr = m.predict_view(D.center_crop(img, m.input_wh))
        P.append([pr[d] for d in shared])
        T.append([float(labels[p].get(d, 0)) for d in shared])
        used.append(p)
    return shared, np.array(P, np.float32), np.array(T, np.float32), used


def calibrate(stamp: str, floor=0.30, min_recall=0.95, log=print) -> dict:
    cfg = D.load_config()
    cur = cfg.get("thresholds", {})
    defects, pv, tv, _ = _scores(stamp, "val_paths")
    _, pt, tt, _ = _scores(stamp, "test_paths")
    new, notes = {}, {}
    for i, d in enumerate(defects):
        t, note = pick(pv[:, i], tv[:, i], floor, min_recall)
        new[d] = cur.get(d, 0.5) if t is None else t
        notes[d] = note
    cur_v = np.array([cur.get(d, 0.5) for d in defects])
    new_v = np.array([new[d] for d in defects])
    res = {"stamp": stamp, "defects": defects, "current": {d: float(cur.get(d, 0.5)) for d in defects}, "new": new,
           "notes": notes, "val": {}, "test": {}}
    for name, (P, T) in (("val", (pv, tv)), ("test", (pt, tt))):
        for label, th in (("current", cur_v), ("new", new_v)):
            r = image_level(P, T, th)
            r["per_defect"] = {}
            pred = P >= th
            for i, d in enumerate(defects):
                tp = int((pred[:, i] & (T[:, i] == 1)).sum())
                fp = int((pred[:, i] & (T[:, i] == 0)).sum())
                r["per_defect"][d] = {"support": int(T[:, i].sum()), "tp": tp, "fp": fp, "fn": int(T[:, i].sum()) - tp}
            res[name][label] = r
    return res


def report(res: dict) -> str:
    out = [f"checkpoint {res['stamp']}", "", f"{'defect':<16}{'current':>8}{'new':>7}   note"]
    for d in res["defects"]:
        mark = "" if abs(res["current"][d] - res["new"][d]) < 1e-9 else "  <- changed"
        out.append(f"{d:<16}{res['current'][d]:>8.2f}{res['new'][d]:>7.2f}   {res['notes'][d]}{mark}")
    for name in ("val", "test"):
        out += ["", f"{name.upper()} ({res[name]['current']['n']} images, {res[name]['current']['good']} good, "
                    f"{res[name]['current']['defective']} defective)"]
        for label in ("current", "new"):
            r = res[name][label]
            fp = sum(v["fp"] for v in r["per_defect"].values())
            fn = sum(v["fn"] for v in r["per_defect"].values())
            out.append(f"  {label:<8} GOOD bottles called defective: {r['good_called_defective']}/{r['good']}   "
                       f"defective bottles passed: {r['defective_passed']}/{r['defective']}   "
                       f"per-defect false alarms {fp}, misses {fn}")
    return "\n".join(out)


def apply(res: dict) -> None:
    cfg = D.load_config()
    cfg["thresholds_before_calibration"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": res["stamp"],
                                            "thresholds": dict(cfg.get("thresholds", {}))}
    cfg["thresholds"] = {**cfg.get("thresholds", {}), **res["new"]}
    D.save_config(cfg)


def restore() -> bool:
    """Put back the thresholds that were in config.json before the last --apply."""
    cfg = D.load_config()
    prev = cfg.get("thresholds_before_calibration")
    if not prev:
        return False
    cfg["thresholds"] = dict(prev["thresholds"])
    cfg.pop("thresholds_before_calibration", None)
    D.save_config(cfg)
    return True


def selftest():
    rng = np.random.RandomState(0)
    neg = rng.beta(2, 20, 200)                       # negatives mostly low, a few noisy
    pos = rng.beta(20, 2, 40)                        # positives high
    prob = np.concatenate([neg, pos])
    truth = np.concatenate([np.zeros(200), np.ones(40)])
    t, note = pick(prob, truth)
    assert 0.3 <= t <= 0.9, (t, note)                # a sensible mid threshold, not 0.05
    # fully saturated validation: the middle of the perfect band, not its extreme end
    sat = pick(np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99]), np.array([0, 0, 0, 1, 1, 1.0]))[0]
    assert 0.4 <= sat <= 0.6, sat
    assert pick(prob, np.zeros_like(truth))[0] is None
    # a noisy negative at 0.2 must not drag a clean problem down to threshold 0.05
    p2 = np.array([0.02, 0.2, 0.9, 0.95]); t2 = np.array([0, 0, 1, 1.0])
    assert pick(p2, t2)[0] >= 0.3
    # recall protection: positives at 0.2 force a lower threshold than the floor
    p3 = np.array([0.01, 0.02, 0.2, 0.25, 0.22, 0.21, 0.23, 0.24]); t3 = np.array([0, 0, 1, 1, 1, 1, 1, 1.0])
    assert pick(p3, t3)[0] < 0.3
    r = image_level(np.array([[0.9, 0.1], [0.1, 0.1], [0.1, 0.8]]), np.array([[0, 0], [0, 0], [0, 1.0]]),
                    np.array([0.5, 0.5]))
    assert r["good_called_defective"] == 1 and r["defective_passed"] == 0
    print("ok  calibrate_thresholds: validation-chosen thresholds (floor, recall guard, no invention without positives), "
          "image-level good-called-defective / defective-passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
        sys.exit(0)
    D.use_project(D.PROJECT)
    stamp = next((x for x in a if not x.startswith("--")), None) or D.load_config().get("active_model")
    if "--restore" in a:
        print("restored the previous thresholds" if restore() else "nothing to restore")
        sys.exit(0)
    res = calibrate(stamp)
    print(report(res))
    if "--apply" in a:
        apply(res)
        print("\napplied to config.json (previous thresholds kept in thresholds_before_calibration)")
