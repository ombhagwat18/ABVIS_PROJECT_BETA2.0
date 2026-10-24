"""Live-camera check of a candidate classifier: does it behave on frames from the REAL cameras as it did on the test set?

    python -m tools.live_check capture [--n 30]              # grab frames from line_cameras (settings.json) -> captures/live_check/<time>/
    python -m tools.live_check score <folder> <project> <stamp>   # bottle outline -> crop -> model; confidence table + sheet
    python -m tools.live_check --selftest

capture uses the driver controls saved in settings.json camera_controls (focus, brightness, white balance) and asks each camera for
3840x2160 first, 1920x1080 if the driver refuses; the size actually delivered is recorded. score uses the auto-labeller's colour+SAM
bottle outline ONLY to cut the crop (a localiser, no label); frames where no trusted outline is found are listed as 'no bottle'.
Read-only on the models; the camera is opened once per camera and released.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _settings() -> dict:
    return json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))


def open_cam(index: int, ctrl: dict):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    got = None
    for w, h in ((3840, 2160), (1920, 1080)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(3):
            cap.read()
        ok, f = cap.read()
        if ok and f is not None and f.shape[1] >= w - 8:
            got = (w, h)
            break
    if ctrl:
        if "autofocus" in ctrl:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, ctrl["autofocus"])
        if "focus" in ctrl:
            cap.set(cv2.CAP_PROP_FOCUS, ctrl["focus"])
        if "auto_wb" in ctrl:
            cap.set(cv2.CAP_PROP_AUTO_WB, ctrl["auto_wb"])
        if "wb_temperature" in ctrl:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, ctrl["wb_temperature"])
        if "brightness" in ctrl:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, ctrl["brightness"])
    return cap, got


def capture(n: int = 30, gap_s: float = 0.4) -> Path:
    s = _settings()
    out = ROOT / "captures" / "live_check" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for i in s.get("line_cameras", [1, 2]):
        ctrl = (s.get("camera_controls") or {}).get(str(i), {})
        cap, size = open_cam(int(i), ctrl)
        for _ in range(20):                                    # auto exposure / white balance settle
            cap.read()
        saved = 0
        for k in range(n):
            ok, f = cap.read()
            if ok and f is not None:
                cv2.imwrite(str(out / f"cam{i}_{k:03d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            time.sleep(gap_s)
        shape = f.shape[:2] if ok and f is not None else None
        info[str(i)] = {"requested": size, "delivered_hw": shape, "frames": saved, "controls": ctrl}
        cap.release()
    (out / "capture.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return out


def reliability_lines(P: np.ndarray) -> dict:
    """Share of all scores per band: where the model's confidence sits on live frames."""
    edges = [0, 0.1, 0.5, 0.9, 1.0001]
    h, _ = np.histogram(P.ravel(), bins=edges)
    return {f"{edges[i]:.1f}-{min(edges[i + 1], 1.0):.1f}": round(float(h[i]) / max(1, P.size), 3) for i in range(4)}


def score(folder: Path, project: str, stamp: str) -> dict:
    from vision import autoseed
    from vision import dataset as D
    from vision import infer
    from tools import make_crops as MC
    D.use_project(project)
    m = infer.Model(stamp)
    thr = json.loads((D.MODELS / stamp / "metrics.json").read_text())["per_defect"]
    thr = {d: v["threshold"] for d, v in thr.items()}
    seeder = autoseed.Seeder(use_detector=False)
    rows, sheet = [], []
    for p in sorted(folder.glob("cam*_*.jpg")):
        im = cv2.imread(str(p))
        im4k = cv2.resize(im, (3840, 2160), interpolation=cv2.INTER_CUBIC) if im.shape[1] < 3000 else im   # the seeder works at half of 4K
        polys = [q for q in seeder.seed(im4k) if q["cls"] == "bottle"]
        if not polys or polys[0]["score"] < 0.8:
            rows.append({"file": p.name, "bottle": False})
            continue
        crop = MC.cut(im, MC.crop_box(polys[0]["points"], im.shape, 288 / 640), (288, 640))
        pr = m.predict(crop)
        flags = [d for d in m.defects if pr[d] >= thr.get(d, 0.5)]
        rows.append({"file": p.name, "bottle": True, "flags": flags, "max": round(max(pr.values()), 3),
                     "scores": {d: round(v, 3) for d, v in pr.items()}})
        if len(sheet) < 24:
            t = cv2.resize(crop, (144, 320))
            cv2.putText(t, (",".join(f[:6] for f in flags) or "GOOD")[:22], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            sheet.append(t)
    withb = [r for r in rows if r["bottle"]]
    P = np.array([[r["scores"][d] for d in m.defects] for r in withb], np.float32) if withb else np.zeros((0, len(m.defects)))
    res = {"folder": str(folder), "frames": len(rows), "no_bottle": len(rows) - len(withb),
           "flagged_any": sum(1 for r in withb if r["flags"]), "good": sum(1 for r in withb if not r["flags"]),
           "score_bands_share": reliability_lines(P) if len(withb) else {},
           "flags_count": {d: sum(1 for r in withb if d in r["flags"]) for d in m.defects}, "rows": rows}
    if sheet:
        cols = 8
        sheet += [np.zeros_like(sheet[0])] * ((-len(sheet)) % cols)
        cv2.imwrite(str(folder / "crops_scored.jpg"), np.vstack([np.hstack(sheet[i:i + cols]) for i in range(0, len(sheet), cols)]))
    (folder / "live_score.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def selftest():
    P = np.array([[0.01, 0.95], [0.3, 0.02]], np.float32)
    b = reliability_lines(P)
    assert abs(sum(b.values()) - 1.0) < 1e-6 and b["0.9-1.0"] == 0.25
    print("live_check self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["capture", "score"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "capture":
        print(capture(a.n))
    elif a.cmd == "score":
        r = score(Path(a.args[0]), a.args[1], a.args[2])
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    else:
        ap.print_help()
