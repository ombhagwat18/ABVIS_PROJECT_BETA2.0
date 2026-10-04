"""Dashboard: label, manage defect types, retrain, watch the line.

One process. Run it, open http://localhost:8000.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

import dataset as D
import infer

app = FastAPI(title="Bottle Inspection")
cam = infer.Camera()

_train_lock = threading.Lock()
_train_log: list[str] = []
_train_busy = False
_reject_log = D.DATA / "rejects.csv"


def _labels():
    return D.load_labels()


# ------------------------------------------------------------------- dashboard

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def state():
    defects, labels = _labels()
    cfg = D.load_config()
    models = infer.list_models()
    active = cfg.get("active_model")
    metrics = None
    if active and (D.MODELS / active / "metrics.json").exists():
        metrics = json.loads((D.MODELS / active / "metrics.json").read_text())
    return {
        "defects": defects,
        "counts": D.counts(defects, labels),
        "config": cfg,
        "models": models,
        "active_model": active,
        "metrics": metrics,
        "training": _train_busy,
        "camera": {"running": cam._thread is not None and cam._thread.is_alive(),
                   "error": cam.error, "source": cam.source,
                   "model": cam.model.stamp if cam.model else None},
    }


@app.get("/api/images")
def images(defect: str = "", mode: str = "all", page: int = 0, per: int = 60, q: str = ""):
    """mode: all | pos | neg | good | inbox   (pos/neg are the +ve/-ve buckets)"""
    defects, labels = _labels()
    items = []
    for p in sorted(labels, key=D._natkey):
        row = labels[p]
        if q and q.lower() not in p.lower():
            continue
        if mode == "inbox" and row.get("reviewed", 1):
            continue
        if mode == "good" and not (row.get("reviewed", 1) and D.is_good(row, defects)):
            continue
        if mode in ("pos", "neg"):
            if defect not in defects:
                continue
            if bool(row.get(defect, 0)) != (mode == "pos"):
                continue
            if mode == "neg" and not row.get("reviewed", 1):
                continue
        items.append({"path": p, "labels": [d for d in defects if row.get(d)],
                      "reviewed": bool(row.get("reviewed", 1))})
    total = len(items)
    return {"total": total, "page": page, "per": per,
            "items": items[page * per:(page + 1) * per]}


@app.get("/thumb/{rel:path}")
def thumb(rel: str):
    b = D.thumbnail(rel)
    if b is None:
        raise HTTPException(404, "no such image")
    return Response(b, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/full/{rel:path}")
def full(rel: str, crop: int = 0):
    """crop=1 shows exactly what the model sees -- catches a bad ROI before it
    costs an hour of training."""
    img = D.imread(D.IMAGE_ROOT / rel)
    if img is None:
        raise HTTPException(404, "no such image")
    cfg = D.load_config()
    if crop:
        img = D.prepare(img, cfg)
    else:
        h, w = img.shape[:2]
        s = 900 / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        if cfg.get("roi"):
            x, y, rw, rh = [round(v * s) for v in cfg["roi"]]
            cv2.rectangle(img, (x, y), (x + rw, y + rh), (90, 220, 90), 2)
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


# --------------------------------------------------------------------- editing

@app.post("/api/label")
def set_label(body: dict = Body(...)):
    """{paths:[...], defect:'tilt_cap'|null, value:0|1, clear_all:bool}

    Always one CSV rewrite, whatever the edit -- 'mark good' must not cost one
    full rewrite per defect column.
    """
    paths, defect, value = body["paths"], body.get("defect"), int(body.get("value", 1))
    clear_all = bool(body.get("clear_all"))
    defects, labels = _labels()
    if defect is not None and defect not in defects:
        raise HTTPException(400, f"unknown defect {defect!r}")
    for p in paths:
        if p not in labels:
            continue
        if clear_all:
            for d in defects:
                labels[p][d] = 0
        elif defect is not None:
            labels[p][defect] = value
        labels[p]["reviewed"] = 1
    D.save_labels(defects, labels)
    return {"ok": True, "n": len(paths), "counts": D.counts(defects, labels)}


@app.post("/api/defect")
def defect_op(body: dict = Body(...)):
    """{op:'add'|'rename'|'delete', name, new_name}. A defect is just a column."""
    op = body["op"]
    defects, labels = _labels()

    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    if op == "add":
        name = slug(body["name"])
        if not name:
            raise HTTPException(400, "empty name")
        if name in defects or name in D.RESERVED:
            raise HTTPException(400, f"{name!r} already exists")
        defects.append(name)
        for row in labels.values():
            row[name] = 0
    elif op == "rename":
        old, new = body["name"], slug(body["new_name"])
        if old not in defects:
            raise HTTPException(400, "no such defect")
        if not new or new in defects or new in D.RESERVED:
            raise HTTPException(400, "bad new name")
        defects[defects.index(old)] = new
        for row in labels.values():
            row[new] = row.pop(old, 0)
    elif op == "delete":
        name = body["name"]
        if name not in defects:
            raise HTTPException(400, "no such defect")
        defects.remove(name)
        for row in labels.values():
            row.pop(name, None)
    else:
        raise HTTPException(400, "op must be add/rename/delete")

    D.save_labels(sorted(defects), labels)
    return {"ok": True, "defects": sorted(defects)}


@app.post("/api/config")
def set_config(body: dict = Body(...)):
    cfg = D.load_config()
    for k in ("roi", "input_wh", "camera", "thresholds"):
        if k in body:
            cfg[k] = body[k]
    D.save_config(cfg)
    return {"ok": True, "config": cfg}


# -------------------------------------------------------------------- training

@app.post("/api/train")
def start_train(body: dict = Body(default={})):
    global _train_busy
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, "training already running")
    _train_log.clear()
    _train_busy = True
    epochs = int(body.get("epochs", 25))

    def work():
        global _train_busy
        try:
            import train
            train.run(epochs=epochs, log=lambda m: _train_log.append(str(m)))
            _train_log.append("__DONE__")
        except Exception as e:
            _train_log.append(f"ERROR: {type(e).__name__}: {e}")
            _train_log.append("__DONE__")
        finally:
            _train_busy = False
            _train_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


@app.get("/api/train/log")
def train_log(since: int = 0):
    return {"lines": _train_log[since:], "n": len(_train_log), "busy": _train_busy}


@app.get("/api/metrics/{stamp}")
def metrics(stamp: str):
    p = D.MODELS / stamp / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "no metrics for that model")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/api/model/activate")
def activate(body: dict = Body(...)):
    """Rollback. Thresholds come from that model's own metrics, not the last run's."""
    stamp = body["stamp"]
    if stamp not in infer.list_models():
        raise HTTPException(404, "no such model")
    cfg = D.load_config()
    cfg["active_model"] = stamp
    mp = D.MODELS / stamp / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        cfg["thresholds"] = {d: v["threshold"] for d, v in m["per_defect"].items()}
    D.save_config(cfg)
    cam.load_model(stamp)
    return {"ok": True, "config": cfg}


# ------------------------------------------------------------------- live view

_cameras_cache: list | None = None
UPLOADS = D.DATA / "uploads"


def _videos() -> list[dict]:
    if not UPLOADS.exists():
        return []
    return [{"name": p.name, "path": str(p), "mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(UPLOADS.iterdir()) if p.is_file()]


@app.get("/api/cameras")
def cameras(refresh: int = 0):
    """Detected webcams plus any uploaded videos. Cached, because probing opens
    every device in turn and cannot run while the stream already owns one."""
    global _cameras_cache
    running = cam._thread is not None and cam._thread.is_alive()
    if _cameras_cache is None or (refresh and not running):
        if running:
            raise HTTPException(409, "stop the camera before scanning for devices")
        _cameras_cache = infer.list_cameras()
    return {"cameras": _cameras_cache, "videos": _videos(),
            "scanning_blocked": bool(running)}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """Take a video from the operator's PC and make it a selectable source."""
    # Trust boundary: the filename comes from the browser. Strip it to a bare
    # name and whitelist the characters, or "../../app.py" lands wherever it likes.
    raw = Path(file.filename or "video.mp4").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw).lstrip(".") or "video.mp4"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / name

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)   # stream, never load it all

    # Check OpenCV can actually decode it now, rather than letting Start fail
    # later with "cannot open camera source". A container it cannot read (some
    # .mkv, HEVC without the codec) is a clear message here and a mystery there.
    cap = infer.open_capture(str(dest))
    ok = cap.isOpened() and cap.read()[0]
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if not ok:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"OpenCV cannot decode {raw!r}. Try an MP4 (H.264).")

    return {"ok": True, "path": str(dest), "name": name,
            "frames": frames if frames > 0 else None,
            "seconds": round(frames / fps, 1) if frames > 0 and fps > 0 else None}


@app.post("/api/upload/delete")
def upload_delete(body: dict = Body(...)):
    p = (UPLOADS / Path(body["name"]).name).resolve()
    if p.parent != UPLOADS.resolve() or not p.exists():
        raise HTTPException(404, "no such upload")
    p.unlink()
    return {"ok": True}


@app.post("/api/camera")
def camera(body: dict = Body(...)):
    if body.get("stop"):
        cam.stop()
        return {"ok": True, "running": False}
    cfg = D.load_config()
    src = body.get("source", cfg.get("camera", 0))
    cfg["camera"] = src
    D.save_config(cfg)
    if cfg.get("active_model"):
        cam.load_model(cfg["active_model"])
    cam.start(src)
    time.sleep(0.8)  # give the capture thread a chance to report a bad source
    if cam.error:
        raise HTTPException(400, cam.error)
    return {"ok": True, "running": True}


@app.get("/stream.mjpg")
def stream():
    def gen():
        while True:
            b = cam.overlay()
            if b is None:
                time.sleep(0.2)
                continue
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/api/live")
def live():
    with cam.lock:
        probs, hits, ok = dict(cam.probs), list(cam.hits), cam.ok
    return {"probs": probs, "hits": hits, "pass": ok, "fps": round(cam.fps, 1),
            "error": cam.error}


@app.post("/api/capture")
def capture(body: dict = Body(default={})):
    """Snapshot straight from the live view into a label. This is the loop that
    fills missing_cap and grows the good-bottle set without a separate session."""
    frame = cam.snapshot()
    if frame is None:
        raise HTTPException(400, "camera is not running")
    D.INBOX.mkdir(parents=True, exist_ok=True)
    rel = f"{D.INBOX_DIR}/cap_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
        str(D.IMAGE_ROOT / rel))

    defects, labels = _labels()  # sync_new_images picks the new file up
    wanted = [d for d in body.get("defects", []) if d in defects]
    labels.setdefault(rel, {d: 0 for d in defects})
    for d in defects:
        labels[rel][d] = 1 if d in wanted else 0
    labels[rel]["reviewed"] = 1 if body.get("reviewed", True) else 0
    D.save_labels(defects, labels)
    return {"ok": True, "path": rel, "labels": wanted,
            "counts": D.counts(defects, labels)}


@app.post("/api/reject-log")
def reject_log(body: dict = Body(...)):
    """Append one line to the shift report."""
    new = not _reject_log.exists()
    D.DATA.mkdir(parents=True, exist_ok=True)
    with _reject_log.open("a", encoding="utf-8", newline="") as f:
        if new:
            f.write("time,verdict,defects,confidences\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                f"{'PASS' if body.get('pass') else 'REJECT'},"
                f"\"{';'.join(body.get('hits', []))}\","
                f"\"{json.dumps(body.get('probs', {}))[1:-1]}\"\n")
    return {"ok": True}


# ------------------------------------------------------------------ duplicates

@app.get("/api/duplicates")
def duplicates(threshold: float = 4.0):
    """Consecutive frames that are near-identical. 319 'Damaged Bottle' images
    may be a dozen real bottles -- that ceiling matters more than any epoch count."""
    _, labels = _labels()
    paths = sorted(labels, key=D._natkey)
    scenes = D.scene_map(paths, threshold)      # same grouping the split uses

    runs: dict[str, list[str]] = {}
    for p in paths:
        runs.setdefault(scenes.get(p, p), []).append(p)

    groups = [{"folder": k.split("#")[0], "n": len(v), "paths": v[:6]}
              for k, v in runs.items() if len(v) > 1]
    groups.sort(key=lambda g: -g["n"])
    return {"groups": groups[:60], "n_images": len(labels),
            "est_distinct_bottles": len(runs)}


if __name__ == "__main__":
    import uvicorn
    D.DATA.mkdir(parents=True, exist_ok=True)
    defects, labels = D.load_labels()
    print(f"{len(labels)} images, {len(defects)} defects: {', '.join(defects)}")
    print(json.dumps(D.counts(defects, labels), indent=2))
    cfg = D.load_config()
    if cfg.get("active_model"):
        try:
            cam.load_model(cfg["active_model"])
            print(f"loaded model {cfg['active_model']}")
        except Exception as e:
            print(f"could not load model: {e}")
    print("\n  ->  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
