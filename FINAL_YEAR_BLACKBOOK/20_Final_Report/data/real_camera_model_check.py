"""Read-only model check: classifiers + detectors on their test sets and on real EMEET frames.
Writes nothing into the project (only into the scratchpad)."""
from __future__ import annotations
import glob, json, os, sys, time
from pathlib import Path

ROOT = Path(r"E:\MACHINE LEARNING PROJECT\ABVIS\ABVIS")
OUT = Path(__file__).resolve().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import cv2, numpy as np, torch
from vision import dataset as D
from vision import infer, train

D.use_project("om_bottle") if hasattr(D, "use_project") else None

REAL = sorted(f for f in glob.glob("captures/*/*.png") if "_overlay" not in f and "survey" not in f
              and "GT_" not in f) + sorted(glob.glob("projects/om_bottle/production/evidence/*/*FAULT.jpg"))[:4]
print("real frames:", len(REAL))

# ------------------------------------------------------------------ classifiers
res = {"classifier": {}, "detector": {}}
stamps = sorted(p.parent.name for p in D.MODELS.glob("*/model.pt"))
for st in stamps:
    m = infer.Model(st)
    thr = {d: v["threshold"] for d, v in json.loads((D.MODELS / st / "metrics.json").read_text())["per_defect"].items()}
    t0 = time.time()
    te = train.evaluate_test(st, log=lambda *a: None)
    rows = []
    for f in REAL:
        img = cv2.imread(f)
        p = m.predict(img)
        hits = [d for d, v in p.items() if v >= thr.get(d, 0.5)]
        rows.append({"frame": f, "hits": hits, "probs": {k: round(v, 3) for k, v in p.items()}})
    res["classifier"][st] = {"arch": m.arch, "test_macro_f1": te["macro_f1"], "n_test": te["n_test"],
                             "thresholds": thr, "real": rows, "secs": round(time.time() - t0, 1)}
    flagged = sum(1 for r in rows if r["hits"])
    print(f"CLS {st} {m.arch:20} test macroF1={te['macro_f1']:.3f} (n={te['n_test']})  real frames flagged defective: {flagged}/{len(rows)}")
    del m; torch.cuda.empty_cache()

# ------------------------------------------------------------------ detectors
from ultralytics import YOLO
dets = {"v8s_v3": "models/stage2_yolo/bench_yolov8s_v3/weights/best.pt",
        "v8n_v3": "models/stage2_yolo/bench_yolov8n_v3/weights/best.pt",
        "stage2_best(v1, old active)": "models_archive/previous/stage2_yolo/stage2_best.pt"}
# portable data.yaml (the exported one points at a folder that no longer exists)
dy = OUT / "data_v3.yaml"
dy.write_text(f"path: {(ROOT / 'stage2_dataset/yolo_export_v3').as_posix()}\ntrain: images/train\nval: images/val\n"
              f"test: images/test\nnc: 3\nnames: ['bottle', 'cap', 'label']\n")
sheet = []
for name, w in dets.items():
    y = YOLO(w)
    v = y.val(data=str(dy), split="test", imgsz=640, batch=4, workers=0, plots=False, verbose=False,
              project=str(OUT / "val"), name=name.split("(")[0], exist_ok=True)
    rows = []
    for f in REAL:
        r = y.predict(f, imgsz=640, conf=0.25, verbose=False, workers=0)[0]
        H, W = r.orig_shape
        boxes = [(r.names[int(c)], round(float(s), 2), [int(x) for x in b], round(float((b[2]-b[0])*(b[3]-b[1])/(W*H)), 2))
                 for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        rows.append({"frame": f, "boxes": boxes})
        if name == "v8s_v3":
            im = r.plot(); h = 300; sheet.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)))
    res["detector"][name] = {"test_map50": round(float(v.box.map50), 3), "test_map50_95": round(float(v.box.map), 3),
                             "real": rows}
    print(f"DET {name:28} test mAP50={v.box.map50:.3f} mAP50-95={v.box.map:.3f}")
    del y; torch.cuda.empty_cache()

(OUT / "model_check.json").write_text(json.dumps(res, indent=1))
rows_img = [np.hstack(sheet[i:i + 4] + [np.zeros_like(sheet[0])] * (4 - len(sheet[i:i + 4]))) for i in range(0, len(sheet), 4)]
cv2.imwrite(str(OUT / "det_sheet.jpg"), np.vstack(rows_img))
print("done")
