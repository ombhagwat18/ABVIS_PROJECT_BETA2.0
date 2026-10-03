"""Camera benchmarking and system performance sampling.

Two jobs that share their measuring tools: benchmark_camera() sweeps a camera
through resolution/FPS settings once and records what it actually delivered,
and system_stats() samples CPU/GPU/memory continuously while the line runs.

Requested FPS is not delivered FPS. A webcam asked for 1920x1080 at 30 will
quietly give 7.5 under its own auto-exposure in poor light, and OpenCV reports
the number you asked for, not the number you got. Everything here is measured
from wall-clock deltas on real frames for that reason.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

try:                                  # optional: system-wide CPU and RAM
    import psutil
except ImportError:                   # degraded, never absent
    psutil = None

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------- system stats

_proc_t0 = (time.process_time(), time.perf_counter())


def system_stats() -> dict:
    """CPU, GPU and memory, as far as this machine will say.

    Process CPU comes from process_time() and needs no dependency; the
    system-wide numbers need psutil. Missing values stay None, never 0.0 -- a
    dashboard showing 0% GPU on a machine with no GPU is telling a lie that
    reads like an idle GPU.
    """
    global _proc_t0
    out: dict = {"cpu_proc": None, "cpu_sys": None, "ram_mb": None, "ram_pct": None,
                 "gpu_name": None, "gpu_mem_mb": None, "gpu_mem_total_mb": None,
                 "gpu_util": None, "gpu_temp": None}

    t_cpu, t_wall = time.process_time(), time.perf_counter()
    d_cpu, d_wall = t_cpu - _proc_t0[0], t_wall - _proc_t0[1]
    if d_wall > 0.05:                       # too short a window is just noise
        out["cpu_proc"] = round(100.0 * d_cpu / d_wall / max(1, os.cpu_count() or 1), 1)
        _proc_t0 = (t_cpu, t_wall)

    if psutil is not None:
        try:
            out["cpu_sys"] = psutil.cpu_percent(interval=None)
            out["ram_pct"] = psutil.virtual_memory().percent
            out["ram_mb"] = round(psutil.Process().memory_info().rss / 1e6)
        except Exception:
            pass

    if torch is not None and torch.cuda.is_available():
        try:
            out["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            out["gpu_mem_mb"] = round((total - free) / 1e6)
            out["gpu_mem_total_mb"] = round(total / 1e6)
        except Exception:
            pass
        try:                                # only present with pynvml installed
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            out["gpu_util"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            out["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
    return out


# -------------------------------------------------------------- image quality

def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is crisper; a motion-blurred or
    out-of-focus frame collapses towards zero."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_quality(frame: np.ndarray) -> dict:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {"brightness": round(float(g.mean()), 1),
            "contrast": round(float(g.std()), 1),
            "sharpness": round(sharpness(g), 1),
            "clipped": round(float((g >= 250).mean() * 100), 2)}


# ----------------------------------------------------------------- the sweep

def fourcc_str(cap) -> str:
    """The FOURCC the driver reports it is delivering ('' if it will not say)."""
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).replace(chr(0), "").strip() if v > 0 else ""


# The comparison set from the architecture doc, plus two lower resolutions
# worth knowing about when the line PC turns out to have no GPU.
DEFAULT_COMBOS = [(1920, 1080, 15), (1920, 1080, 20), (1920, 1080, 30),
                  (1280, 720, 30), (640, 480, 30)]


def benchmark_camera(index: int, combos=None, seconds: float = 3.0,
                     model=None, progress=print, should_stop=None, fourcc: str | None = "MJPG") -> list[dict]:
    """Run one camera through each (width, height, fps) and measure what it did.

    `model` is an optional infer.Model; when given, every fifth frame is also
    scored, so inference time is measured on the frames this configuration
    actually produces rather than on a stand-in.

    `fourcc` is requested before the size, the same way the line opens its
    cameras (infer.open_capture). Without MJPG a USB 2.0 webcam falls back to
    uncompressed YUY2, which cannot carry 1920x1080 at 30 fps -- the sweep
    would then report the bus limit, not what the line will get.
    """
    import infer

    combos = combos or DEFAULT_COMBOS
    rows = []
    for w, h, fps in combos:
        if should_stop is not None and should_stop():
            break
        progress(f"{w}x{h} @ {fps} fps requested...")
        cap = infer.open_capture(index, (w, h), fourcc)
        if not cap.isOpened():
            cap.release()
            rows.append({"requested": f"{w}x{h}@{fps}", "error": "cannot open camera"})
            continue
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):                  # let auto-exposure settle first
            cap.read()

        lat, quality, infer_ms, n = [], [], [], 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            t = time.perf_counter()
            got, frame = cap.read()
            if not got or frame is None:
                break
            lat.append((time.perf_counter() - t) * 1000)
            n += 1
            if n % 5 == 1:                  # quality every 5th frame; it is not free
                quality.append(frame_quality(frame))
                if model is not None:
                    ti = time.perf_counter()
                    model.predict(frame)
                    infer_ms.append((time.perf_counter() - ti) * 1000)
        elapsed = time.perf_counter() - t0
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fourcc = fourcc_str(cap)
        cap.release()

        row = {"requested": f"{w}x{h}@{fps}",
               "actual_size": f"{actual_w}x{actual_h}", "fourcc": got_fourcc,
               "size_ok": (actual_w, actual_h) == (w, h),
               "frames": n,
               "fps": round(n / elapsed, 1) if elapsed else 0.0,
               "latency_ms": round(float(np.mean(lat)), 1) if lat else None,
               "latency_p95_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
               "jitter_ms": round(float(np.std(lat)), 1) if lat else None}
        for k in ("brightness", "contrast", "sharpness", "clipped"):
            row[k] = round(float(np.mean([q[k] for q in quality])), 1) if quality else None
        row["infer_ms"] = round(float(np.mean(infer_ms)), 1) if infer_ms else None
        if row["infer_ms"]:
            row["max_fps_model"] = round(1000.0 / row["infer_ms"], 1)
        rows.append(row)
        progress(f"  -> {row['actual_size']} at {row['fps']} fps, "
                 f"latency {row['latency_ms']} ms, sharpness {row['sharpness']}")
    return rows


_PROPS = {"auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "exposure": cv2.CAP_PROP_EXPOSURE,
          "autofocus": cv2.CAP_PROP_AUTOFOCUS, "focus": cv2.CAP_PROP_FOCUS, "gain": cv2.CAP_PROP_GAIN,
          "brightness_setting": cv2.CAP_PROP_BRIGHTNESS, "auto_wb": cv2.CAP_PROP_AUTO_WB,
          "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE, "fps_reported": cv2.CAP_PROP_FPS}


def _measure(caps: dict, seconds: float, keep: dict) -> dict:
    """Read every open capture in turn for `seconds`; per source: frames, fps, quality, last frame."""
    n = {k: 0 for k in caps}
    q = {k: [] for k in caps}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for k, cap in caps.items():
            got, frame = cap.read()
            if got and frame is not None:
                n[k] += 1
                keep[k] = frame
                if n[k] % 10 == 1:
                    q[k].append(frame_quality(frame))
    el = time.perf_counter() - t0
    out = {}
    for k in caps:
        out[k] = {"frames": n[k], "fps": round(n[k] / el, 1),
                  **({m: round(float(np.mean([x[m] for x in q[k]])), 1) for m in q[k][0]} if q[k] else {})}
    return out


def survey(indices, size=(1920, 1080), fourcc="MJPG", seconds=4.0, out_dir=None, progress=print) -> dict:
    """What each camera really delivers in the line's capture mode, alone and all together.

    Per camera: DirectShow name, delivered size / FOURCC / fps, the driver's exposure, focus, gain
    and white-balance properties (as reported; -1/0 often means "not exposed by this driver"),
    frame quality, and one saved full-resolution frame. Then all cameras opened at once at
    `size`, 1280x720 and 640x480, because USB bandwidth is shared and one camera alone
    proves nothing about two.
    """
    import json
    import infer
    names = infer.camera_names()
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "requested": f"{size[0]}x{size[1]} {fourcc}",
           "alone": {}, "together": {}}
    for i in indices:
        progress(f"camera {i} alone...")
        t = time.perf_counter()
        cap = infer.open_capture(i, size, fourcc)
        r = {"index": i, "name": names[i] if i < len(names) else "", "opened": bool(cap.isOpened()),
             "open_s": round(time.perf_counter() - t, 2)}
        if r["opened"]:
            for _ in range(15):                      # auto-exposure settles
                cap.read()
            keep = {}
            r.update(_measure({i: cap}, seconds, keep)[i])
            r["size"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            r["fourcc"] = fourcc_str(cap)
            r["props"] = {k: round(float(cap.get(p)), 2) for k, p in _PROPS.items()}
            r["controls"] = infer.applied_controls.get(str(i))      # settings camera_controls: [asked, got]
            r["rotate"] = infer.camera_controls(i).get("rotate") or 0
            if i in keep:
                keep[i] = infer.orient(keep[i], r["rotate"])
            if i in keep:
                r["frame_shape"] = list(keep[i].shape)
                if out_dir:
                    cv2.imwrite(str(out_dir / f"cam{i}_alone.png"), keep[i])
        cap.release()
        rep["alone"][str(i)] = r
        progress(f"  {r}")
    if len(indices) > 1:
        for wh in (tuple(size), (1280, 720), (640, 480)):
            progress(f"all together at {wh[0]}x{wh[1]}...")
            caps = {i: infer.open_capture(i, wh, fourcc) for i in indices}
            try:
                for _ in range(10):
                    for c in caps.values():
                        c.read()
                keep = {}
                m = _measure(caps, seconds, keep)
                for i, c in caps.items():
                    m[i]["size"] = f"{int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                    if out_dir and i in keep:
                        cv2.imwrite(str(out_dir / f"cam{i}_together_{wh[0]}x{wh[1]}.png"), keep[i])
            finally:
                for c in caps.values():
                    c.release()
            rep["together"][f"{wh[0]}x{wh[1]}"] = {str(k): v for k, v in m.items()}
            progress(f"  {m}")
    if out_dir:
        (out_dir / "survey.json").write_text(json.dumps(rep, indent=2))
    return rep


def best_combo(rows: list[dict]) -> dict | None:
    """The configuration to actually run.

    Sharpness first, not frame rate. A defect the camera cannot resolve is
    invisible at any frame rate, and this line inspects a meniscus line and
    label print. Only rows that delivered the resolution they were asked for and
    held at least 80% of their requested frame rate are eligible -- a camera
    that silently dropped to 7 fps is not a configuration, it is a fault.
    """
    ok = []
    for r in rows:
        if r.get("error") or not r.get("size_ok") or not r.get("sharpness"):
            continue
        try:
            want = float(r["requested"].split("@")[1])
        except (KeyError, IndexError, ValueError):
            continue
        if r.get("fps", 0) >= 0.8 * want:
            ok.append(r)
    return max(ok, key=lambda r: r["sharpness"]) if ok else None


def demo():
    """Self-check on the measuring, not on any particular camera."""
    rng = np.random.default_rng(0)
    crisp = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    blur = cv2.GaussianBlur(crisp, (9, 9), 0)
    assert sharpness(crisp) > sharpness(blur) * 2, "blur must score well below crisp"

    dark = np.full((40, 40, 3), 10, np.uint8)
    q = frame_quality(dark)
    assert q["brightness"] < 20 and q["clipped"] == 0.0, q
    assert frame_quality(np.full((40, 40, 3), 255, np.uint8))["clipped"] == 100.0

    s = system_stats()
    assert set(s) >= {"cpu_proc", "ram_mb", "gpu_name"}, s
    assert s["cpu_proc"] is None or 0 <= s["cpu_proc"] <= 100, s

    rows = [{"requested": "1920x1080@30", "size_ok": True, "fps": 29.4, "sharpness": 90.0},
            {"requested": "1920x1080@15", "size_ok": True, "fps": 15.0, "sharpness": 140.0},
            {"requested": "640x480@30", "size_ok": True, "fps": 30.0, "sharpness": 300.0},
            {"requested": "1920x1080@20", "size_ok": False, "fps": 20.0, "sharpness": 999.0}]
    assert best_combo(rows)["requested"] == "640x480@30", "picked on something other than sharpness"
    assert best_combo([{"requested": "1x1@30", "error": "x"}]) is None
    # a camera that could not hold its frame rate is not a usable configuration,
    # however sharp the few frames it did manage were
    assert best_combo([{"requested": "1920x1080@30", "size_ok": True,
                        "fps": 7.5, "sharpness": 999.0}]) is None
    print("ok  quality metrics ordered, best_combo picks on sharpness"
          + ("" if psutil else "  (psutil absent: system CPU/RAM unavailable)"))


if __name__ == "__main__":
    import sys
    if "--survey" in sys.argv:
        # python bench.py --survey 2 3   -> captures/survey_<stamp>/ (frames + survey.json)
        idx = [int(a) for a in sys.argv[sys.argv.index("--survey") + 1:] if a.isdigit()]
        survey(idx, out_dir=Path("captures") / time.strftime("survey_%Y%m%d-%H%M%S"))
    else:
        demo()
