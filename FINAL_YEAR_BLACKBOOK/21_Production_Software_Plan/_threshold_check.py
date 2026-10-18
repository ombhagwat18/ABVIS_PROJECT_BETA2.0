"""Run from the repo root: python FINAL_YEAR_BLACKBOOK/21_Production_Software_Plan/_threshold_check.py <out.json>
Re-score every classifier on its own test set and every detector on the v3 test split; read-only for the project.
Writes one JSON (argv[1])."""
import csv, json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from vision import dataset as D, infer

OUT = Path(sys.argv[1])
REJECT_MIN, PASS_MAX = 0.90, 0.10
S = json.loads((ROOT / "settings.json").read_text())
CFG = json.loads((ROOT / "projects/om_bottle/config.json").read_text())


def live_thr(base):
    shift, ovr = float(S.get("threshold_shift", 0)), S.get("threshold_overrides") or {}
    return {k: v if v > 1.0 else min(0.99, max(0.02, float(ovr.get(k, v + shift)))) for k, v in base.items()}


def bottle_eval(P, Y, defects, thr):
    """two-zone with thr; three-zone with REJECT_MIN / PASS_MAX; only enabled defects (thr <= 1)."""
    en = [i for i, d in enumerate(defects) if thr[d] <= 1.0]
    good = Y[:, en].sum(1) == 0
    fired = (P[:, en] >= np.array([thr[defects[i]] for i in en])).any(1)
    rej3 = (P[:, en] >= REJECT_MIN).any(1)
    pas3 = (P[:, en] <= PASS_MAX).all(1)
    uns3 = ~rej3 & ~pas3
    ng, nd = int(good.sum()), int((~good).sum())
    return {"n_good": ng, "n_defective": nd,
            "two_zone": {"escaped": int((~good & ~fired).sum()), "false_reject": int((good & fired).sum())},
            "three_zone": {"escaped(PASS)": int((~good & pas3).sum()), "false_reject(REJECT)": int((good & rej3).sum()),
                           "unsure": int(uns3.sum()), "good_unsure": int((good & uns3).sum()),
                           "defective_unsure": int((~good & uns3).sum()),
                           "good_pass": int((good & pas3).sum()), "defective_reject": int((~good & rej3).sum())}}


def classifiers():
    res = {}
    for mdir in sorted((D.MODELS).glob("*/metrics.json")):
        stamp = mdir.parent.name
        m = json.loads(mdir.read_text())
        rows = {r[0]: r for r in csv.reader(open(mdir.parent / "labels.csv", encoding="utf-8"))}
        head = rows.pop("path")
        model = infer.Model(stamp)
        defects = model.defects
        P, Y = [], []
        for p in m["test_paths"]:
            img = cv2.imread(str(D.IMAGE_ROOT / p))
            if img is None or p not in rows:
                continue
            pr = model.predict(img)
            P.append([pr[d] for d in defects])
            Y.append([int(rows[p][head.index(d)]) for d in defects])
        P, Y = np.array(P), np.array(Y)
        val_thr = {d: float(m["per_defect"].get(d, {}).get("threshold", 1.01)) for d in defects}
        per = {}
        for i, d in enumerate(defects):
            pos, neg = P[Y[:, i] == 1, i], P[Y[:, i] == 0, i]
            t = val_thr[d]
            per[d] = {"val_threshold": t, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                      "pos_median": round(float(np.median(pos)), 3) if len(pos) else None,
                      "pos_min": round(float(pos.min()), 3) if len(pos) else None,
                      "neg_p95": round(float(np.percentile(neg, 95)), 3), "neg_max": round(float(neg.max()), 3),
                      "recall_at_thr": round(float((pos >= t).mean()), 3) if len(pos) and t <= 1 else None,
                      "false_alarm_at_thr": round(float((neg >= t).mean()), 3) if t <= 1 else None,
                      "recall_at_0.90": round(float((pos >= REJECT_MIN).mean()), 3) if len(pos) else None,
                      "neg_above_0.10": round(float((neg > PASS_MAX).mean()), 3)}
        r = {"arch": model.arch, "n_test": int(len(P)), "temperature": model.temperature, "per_defect": per,
             "bottle_val_thresholds": bottle_eval(P, Y, defects, val_thr)}
        if stamp == CFG.get("active_model"):
            r["config_thresholds"] = CFG["thresholds"]
            r["live_thresholds"] = live_thr(CFG["thresholds"])
            r["bottle_config_thresholds"] = bottle_eval(P, Y, defects, CFG["thresholds"])
            r["bottle_live_thresholds"] = bottle_eval(P, Y, defects, r["live_thresholds"])
        res[stamp] = r
        print(stamp, model.arch, "done", flush=True)
        del model
    return res


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def detectors():
    from ultralytics import YOLO
    imgs = sorted((ROOT / "stage2_dataset/yolo_export_v3/images/test").glob("*"))
    lab = ROOT / "stage2_dataset/yolo_export_v3/labels/test"
    names = ["bottle", "cap", "label"]
    out = {}
    for tag, w in (("stage2_best (configured active)", "models/stage2_yolo/stage2_best.pt"),
                   ("YOLOv8n v3 candidate", "models/stage2_yolo/bench_yolov8n_v3/weights/best.pt"),
                   ("YOLOv8s v3 candidate", "models/stage2_yolo/bench_yolov8s_v3/weights/best.pt")):
        model = YOLO(str(ROOT / w))
        tp_conf = {n: [] for n in names}
        fp_conf = {n: [] for n in names}
        n_gt = {n: 0 for n in names}
        for im in imgs:
            img = cv2.imread(str(im)); H, W = img.shape[:2]
            gt = []
            lf = lab / (im.stem + ".txt")
            for line in (lf.read_text().split("\n") if lf.exists() else []):
                if line.strip():
                    c, x, y, w_, h_ = map(float, line.split())
                    gt.append((int(c), [(x - w_ / 2) * W, (y - h_ / 2) * H, (x + w_ / 2) * W, (y + h_ / 2) * H]))
                    n_gt[names[int(c)]] += 1
            r = model.predict(img, conf=0.05, imgsz=640, verbose=False, device=0)[0]
            dets = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda d: -d[1])
            used = set()
            for c, cf, box in dets:
                c = int(c); best, bj = 0, None
                for j, (gc, gb) in enumerate(gt):
                    if gc == c and j not in used:
                        v = iou(box, gb)
                        if v > best:
                            best, bj = v, j
                if best >= 0.5:
                    used.add(bj); tp_conf[names[c]].append(cf)
                else:
                    fp_conf[names[c]].append(cf)
        per = {}
        for n in names:
            tp, fp = np.array(tp_conf[n]), np.array(fp_conf[n])
            per[n] = {"n_gt": n_gt[n], "tp_conf_median": round(float(np.median(tp)), 3) if len(tp) else None,
                      "tp_conf_min": round(float(tp.min()), 3) if len(tp) else None}
            for t in (0.20, 0.25, 0.40, 0.55, 0.70, 0.80):
                per[n][f"recall@{t}"] = round(float((tp >= t).sum() / max(1, n_gt[n])), 3)
                per[n][f"false_boxes@{t}"] = int((fp >= t).sum())
        out[tag] = {"weights": w, "n_images": len(imgs), "per_class": per}
        print(tag, "done", flush=True)
    return out


def real_camera():
    r = json.loads((ROOT / "FINAL_YEAR_BLACKBOOK/20_Final_Report/data/real_camera_model_check.json").read_text())
    out = {"classifier": {}, "detector": {}}
    for stamp, c in r["classifier"].items():
        pr = [f["probs"] for f in c["real"] if f.get("probs")]
        mx = [max(p.values()) for p in pr]
        out["classifier"][stamp] = {"arch": c["arch"], "frames": len(pr),
                                    "confident_pass(all<=0.10)": sum(1 for p in pr if max(p.values()) <= PASS_MAX),
                                    "confident_reject(any>=0.90)": sum(1 for v in mx if v >= REJECT_MIN),
                                    "unsure": sum(1 for v in mx if PASS_MAX < v < REJECT_MIN),
                                    "median_max_score": round(float(np.median(mx)), 3) if mx else None}
    for tag, d in r["detector"].items():
        bc = [max([b[1] for b in f["boxes"] if b[0] == "bottle"] or [0]) for f in d["real"]]
        cc = [max([b[1] for b in f["boxes"] if b[0] == "cap"] or [0]) for f in d["real"]]
        out["detector"][tag] = {"frames": len(bc), "bottle_conf_median": round(float(np.median(bc)), 3),
                                "frames_bottle>=0.80": sum(1 for v in bc if v >= 0.80),
                                "frames_bottle>=0.25": sum(1 for v in bc if v >= 0.25),
                                "cap_conf_median": round(float(np.median(cc)), 3),
                                "frames_cap>=0.70": sum(1 for v in cc if v >= 0.70)}
    return out


res = {"rules": {"reject_min": REJECT_MIN, "pass_max": PASS_MAX}, "classifiers": classifiers(),
       "detectors": detectors(), "real_camera": real_camera(),
       "settings_det_min_conf": (S.get("decision_rules") or {}).get("det_min_conf_by_class")}
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
