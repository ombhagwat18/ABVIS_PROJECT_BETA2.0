"""System report: MEASURE what this installation really is and compare it with the project specification.

    system_report.bat                                  # double-click (asks before the read-only PLC test)
    python -m tools.system_report                      # computer, cameras, ladder, settings, models, data
    python -m tools.system_report --plc                # + read-only PLC link test on the configured COM port
    python -m tools.system_report --plc --port COM5    # + the same on another port (settings are not changed)
    python -m tools.system_report --no-cameras --no-models   # quick (no device opened, no model loaded)
    python -m tools.system_report --selftest

Writes reports/system/<date-time>/system_report.md + .json (+ one frame per camera). Read-only for the project:
it never writes settings.json, config.json, labels or models, and never writes to the PLC (PLCService link test
and status reads only; no M0 / M1, no Y). Close the application first: a camera or a COM port can be open in one
program only.

Every value is labelled by where it came from: MEASURED (this run), SAVED (settings / files, may be stale),
or NOT MEASURED. The SPEC table below is the project's target - edit it there, not in the report.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL, NOT = "OK", "WARN", "FAIL", "NOT MEASURED"

# ------------------------------------------------------------------------------------------------ specification
# The project's targets. (id, what, required, how it is judged). Edit here; the report quotes this table.
SPEC = {
    "python": "3.8.x 64-bit (code targets Python 3.8)",
    "torch": "2.4.x with CUDA available",
    "gpu_vram_gb": 4.0,            # >= : training and two-camera inference were sized for a 4 GB RTX 3050
    "ram_gb": 16.0,                # >= : below this, training exhausts the paging file
    "disk_free_gb": (20.0, 5.0),   # WARN below the first, FAIL below the second (evidence images, checkpoints)
    "line_cameras": 2,             # cameras the line uses (settings line_cameras)
    "cam_alone_fps": 25.0,         # each camera alone at 1920x1080 MJPG
    "cam_together_fps": 20.0,      # each camera while BOTH stream (shared USB bandwidth), at the line resolution
    "line_resolution": (1920, 1080),
    "plc_replies": 20,             # read-only heartbeats that must all answer
    "plc_latency_ms": 100.0,       # median heartbeat latency (9600 baud Modbus ASCII)
    "classifier_ms": 100.0,        # mean per frame on this computer
    "detector_ms": 100.0,
    "speed_spread_pct": 5.0,       # saved belt-speed calibration: max spread between runs
}


def _r(v, n=1):
    return None if v is None else round(float(v), n)


def _ver(mod: str):
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "?"))
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ sections
def computer() -> dict:
    out = {"os": f"{platform.system()} {platform.release()} ({platform.version()})", "machine": platform.machine(),
           "cpu": platform.processor() or "?", "cpu_threads": os.cpu_count(),
           "python": platform.python_version(), "python_bits": 64 if sys.maxsize > 2 ** 32 else 32,
           "packages": {m: _ver(m) for m in ("torch", "torchvision", "cv2", "numpy", "PIL", "customtkinter",
                                               "ultralytics", "serial", "psutil", "comtypes", "PyQt5")}}
    out["ram_gb"] = None
    try:
        import psutil
        out["ram_gb"] = _r(psutil.virtual_memory().total / 1e9)
    except Exception:
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                out["ram_gb"] = _r(m.total / 1e9)
    du = shutil.disk_usage(str(ROOT))
    out["disk_free_gb"], out["disk_total_gb"] = _r(du.free / 1e9), _r(du.total / 1e9)
    out["cuda"], out["gpu"], out["gpu_vram_gb"] = False, None, None
    try:
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
        out["torch_cuda"] = torch.version.cuda
        if out["cuda"]:
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_vram_gb"] = _r(torch.cuda.get_device_properties(0).total_memory / 1e9)
    except Exception:
        pass
    return out


def serial_ports(settings: dict) -> dict:
    from plc import serial_ports as sp
    ports = [{"port": p, "description": d, "bluetooth": "bluetooth" in d.lower()} for p, d in sp()]
    want = str(settings.get("plc_com") or "")
    return {"ports": ports, "configured": want, "configured_present": any(p["port"] == want for p in ports),
            "configured_is_bluetooth": any(p["port"] == want and p["bluetooth"] for p in ports),
            "usb_serial_candidates": [p["port"] for p in ports if not p["bluetooth"]]}


def plc_link(settings: dict, port: str | None) -> dict:
    """Read-only, through PLCService (the one owner of the link): connect, 20 heartbeats, one status read."""
    from plc import PLCClient, PLCService, SerialTransport
    port = port or str(settings.get("plc_com") or "")
    out = {"port": port, "baud": int(settings.get("plc_baud", 9600)), "format": str(settings.get("plc_format", "7E1")),
           "station": int(settings.get("plc_station", 1)), "source": "MEASURED", "error": None}
    if not port:
        out.update(error="no COM port configured", source=NOT)
        return out
    svc = PLCService(PLCClient(SerialTransport(port, out["baud"], out["format"]), station=out["station"], timeout=1.0),
                     poll_s=0.05, status_period_s=0.5)
    svc.start()
    try:
        out["connect_ms"] = _r(svc.connect())
        out["link_test"] = svc.link_test(SPEC["plc_replies"])
        st = svc.read_status()
        out["plc_run"] = st.get("plc_run")
        out["devices"] = {k: st.get(k) for k in ("inputs", "internal", "outputs", "timers", "counters")}
    except Exception as e:                                    # noqa: BLE001 - reported, never raised
        msg = f"{type(e).__name__}: {e}"
        if "denied" in msg.lower():
            msg += "  (another program owns the port: close ISPSoft / COMMGR / NexaInspect, one program per port)"
        out["error"] = msg
    finally:
        svc.stop()
    return out


def ladder(settings: dict) -> dict:
    from plc import ladder_check as LC
    p = LC.DEFAULT_ISP
    if not p.is_file():
        return {"file": str(p), "error": "saved ladder not found", "source": NOT}
    nets, _ = LC.load(p)
    _, t0 = LC._timer(nets, "T0")
    _, t1 = LC._timer(nets, "T1")
    rows = LC.check(nets, settings.get("plc_t0_s"), settings.get("plc_t1_s"), str(settings.get("estop_device") or ""))
    return {"file": str(p.relative_to(ROOT)), "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
            "networks": len(nets), "T0_preset": t0, "T1_preset": t1,
            "T0_s": _r(int(str(t0).lstrip("K")) / 10, 2) if t0 else None,
            "T1_s": _r(int(str(t1).lstrip("K")) / 10, 2) if t1 else None,
            "requirements": [{"id": r[0], "status": r[1], "requirement": r[2], "found": r[3]} for r in rows],
            "source": "SAVED FILE (the program loaded in the PLC is not readable over Modbus: compare it in ISPSoft)"}


def line_config(settings: dict) -> dict:
    s = settings
    speed, dist = float(s.get("conveyor_mm_s") or 0), float(s.get("inspection_to_reject_mm") or 0)
    t0, t1 = float(s.get("plc_t0_s") or 0), float(s.get("plc_t1_s") or 0)
    cal = s.get("speed_calibration") or None
    out = {"conveyor_mm_s": speed or None, "inspection_to_reject_mm": dist or None,
           "speed_source": ("MEASURED (calibration " + str(cal.get("when") or cal.get("time") or cal.get("date") or "?") + ")"
                            if cal else ("SAVED, no calibration record" if speed else NOT)),
           "speed_calibration": cal, "plc_t0_s": t0, "plc_t1_s": t1,
           "inspect_window_s": s.get("inspect_window_s"), "inspect_frames": s.get("inspect_frames"),
           "line_cameras": s.get("line_cameras") or [], "camera_stations": s.get("camera_stations") or {},
           "fault_action": s.get("fault_action", "REJECT"), "monitor_only": bool(s.get("monitor_only")),
           "threshold_shift": s.get("threshold_shift", 0), "threshold_overrides": s.get("threshold_overrides") or {},
           "det_min_conf_by_class": (s.get("decision_rules") or {}).get("det_min_conf_by_class"),
           "plc_mode": s.get("plc_mode"), "plc_com": s.get("plc_com"), "estop_device": s.get("estop_device") or None}
    if speed and dist:
        travel = dist / speed
        out["travel_s"] = _r(travel, 2)
        out["reject_dispatch_after_trigger_s"] = _r(travel - t0, 2)
        out["reject_cycle_blocks_trigger_s"] = _r(travel + t1, 2)     # M2 busy until M1 sent, then M1 held T0 + T1
    return out


def cameras(settings: dict, folder: Path, seconds: float) -> dict:
    from vision import bench, infer
    names = infer.camera_names()
    found = infer.list_cameras(int(settings.get("camera_probe", 5)))
    line = [int(i) for i in (settings.get("line_cameras") or [])]
    idx = [i for i in line if any(f["index"] == i for f in found)] or [f["index"] for f in found][:2]
    rep = {"directshow_names": names, "found": found, "line_cameras_saved": line, "measured": idx, "source": "MEASURED"}
    if idx:
        rep["survey"] = bench.survey(idx, size=SPEC["line_resolution"], seconds=seconds, out_dir=folder,
                                     progress=lambda m: print("   ", m[:150]))
    return rep


def models(settings: dict, n: int) -> dict:
    import numpy as np
    from vision import dataset as D
    out = {"classifier": None, "detector": None}
    cfg = D.load_config()
    stamp = cfg.get("active_model")
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    c = {"active_model": stamp, "file": None}
    if stamp and (D.MODELS / stamp / "model.pt").is_file():
        f = D.MODELS / stamp / "model.pt"
        c.update(file=str(f.relative_to(ROOT)), size_mb=_r(f.stat().st_size / 1e6, 1))
        try:
            from vision import infer
            m = infer.Model(stamp)
            m.predict(frame)                                          # warm-up (CUDA init)
            t = []
            for _ in range(n):
                t0 = time.perf_counter(); m.predict(frame); t.append((time.perf_counter() - t0) * 1000)
            t.sort()
            c.update(arch=m.arch, input_wh=list(m.input_wh), device=m.dev, defects=m.defects,
                     thresholds=cfg.get("thresholds"), ms_mean=_r(sum(t) / len(t)), ms_p95=_r(t[int(0.95 * (n - 1))]),
                     source="MEASURED")
        except Exception as e:                                        # noqa: BLE001
            c["error"] = f"{type(e).__name__}: {e}"
    else:
        c["error"] = "active classifier checkpoint not found"
    out["classifier"] = c
    from vision import detect
    w = Path(settings.get("detector_weights") or detect.DEFAULT_WEIGHTS)
    w = w if w.is_absolute() else ROOT / w
    d = {"weights": str(w.relative_to(ROOT)) if str(w).startswith(str(ROOT)) else str(w), "present": w.is_file()}
    if w.is_file():
        d["size_mb"] = _r(w.stat().st_size / 1e6, 1)
        d["sha256_12"] = hashlib.sha256(w.read_bytes()).hexdigest()[:12]
        try:
            det = detect.YoloDetector(weights=w)
            t = []
            for _ in range(n):
                t0 = time.perf_counter(); det.detect(frame); t.append((time.perf_counter() - t0) * 1000)
            t.sort()
            d.update(classes=list(det.class_names), device=str(det.device), imgsz=det.imgsz, model_id=det.model_id,
                     ms_mean=_r(sum(t) / len(t)), ms_p95=_r(t[int(0.95 * (n - 1))]), source="MEASURED")
        except Exception as e:                                        # noqa: BLE001
            d["error"] = f"{type(e).__name__}: {e}"
    out["detector"] = d
    return out


def data() -> dict:
    from vision import dataset as D
    out = {"project": D.PROJECT, "labels_csv": None}
    if D.LABELS_CSV.is_file():
        rows = list(csv.reader(open(D.LABELS_CSV, encoding="utf-8")))
        head, body = rows[0], rows[1:]
        rev = [r for r in body if r and r[-1] == "1"]
        out.update(labels_csv=str(D.LABELS_CSV.relative_to(ROOT)), images=len(body), reviewed=len(rev),
                   unreviewed=len(body) - len(rev),
                   good=sum(1 for r in rev if all(x == "0" for x in r[1:-1])),
                   per_defect={c: sum(1 for r in rev if r[i] == "1") for i, c in enumerate(head[1:-1], 1)})
    db = D.PROJECT_DIR / "production" / "production.db"
    if db.is_file():
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            out["production"] = {"runs": con.execute("select count(*) from runs").fetchone()[0],
                                 "bottles": dict(con.execute("select final, count(*) from inspections group by final"))}
        finally:
            con.close()
    return out


# ------------------------------------------------------------------------------------------------ judgement
def judge(rep: dict) -> list:
    """[(item, required, measured, status)] - the specification check."""
    S, rows = SPEC, []
    add = lambda *r: rows.append(r)                                    # noqa: E731
    c = rep.get("computer") or {}
    if c:
        add("Python", S["python"], f"{c['python']} {c['python_bits']}-bit",
            OK if c["python"].startswith("3.8") and c["python_bits"] == 64 else WARN)
        tv = c["packages"].get("torch")
        add("PyTorch + CUDA", S["torch"], f"{tv} | CUDA {'yes' if c.get('cuda') else 'no'}",
            OK if tv and tv.startswith("2.4") and c.get("cuda") else WARN)
        add("GPU memory", f">= {S['gpu_vram_gb']} GB", f"{c.get('gpu') or 'no CUDA GPU'} {c.get('gpu_vram_gb') or ''}",
            OK if (c.get("gpu_vram_gb") or 0) >= S["gpu_vram_gb"] - 0.3 else WARN)
        add("RAM", f">= {S['ram_gb']} GB", f"{c.get('ram_gb')} GB",
            NOT if c.get("ram_gb") is None else OK if c["ram_gb"] >= S["ram_gb"] - 0.6 else WARN)
        w, f = S["disk_free_gb"]
        add("Free disk", f">= {w} GB", f"{c['disk_free_gb']} GB",
            OK if c["disk_free_gb"] >= w else WARN if c["disk_free_gb"] >= f else FAIL)
    sp = rep.get("serial")
    if sp:
        add("PLC COM port present", "configured port exists, not Bluetooth",
            f"{sp['configured'] or '-'} ({'present' if sp['configured_present'] else 'NOT present'}"
            f"{', Bluetooth' if sp['configured_is_bluetooth'] else ''}); ports: "
            + (", ".join(p["port"] for p in sp["ports"]) or "none"),
            OK if sp["configured_present"] and not sp["configured_is_bluetooth"] else FAIL)
    pl = rep.get("plc")
    if pl is None:
        add("PLC link (read-only)", f"{S['plc_replies']}/{S['plc_replies']} replies, median <= {S['plc_latency_ms']} ms, RUN",
            "not tested (run with --plc)", NOT)
    elif pl.get("error"):
        add("PLC link (read-only)", f"{S['plc_replies']} replies", pl["error"][:120], FAIL)
    else:
        lt = pl["link_test"]
        good = lt["replies"] == S["plc_replies"] and lt["median_ms"] <= S["plc_latency_ms"] and lt["plc_run"]
        add("PLC link (read-only)", f"{S['plc_replies']}/{S['plc_replies']} replies, median <= {S['plc_latency_ms']} ms, RUN",
            f"{lt['replies']} replies, median {lt['median_ms']:.0f} ms, max {lt['max_ms']:.0f} ms, "
            f"PLC {'RUN' if lt['plc_run'] else 'STOP'} on {pl['port']}", OK if good else WARN)
    ld = rep.get("ladder") or {}
    lc = rep.get("line") or {}
    if ld.get("T0_s") is not None:
        same = abs(ld["T0_s"] - (lc.get("plc_t0_s") or 0)) < 1e-6 and abs(ld["T1_s"] - (lc.get("plc_t1_s") or 0)) < 1e-6
        add("T0 / T1 (saved ladder vs settings)", "equal; loaded PLC program checked in ISPSoft",
            f"ladder {ld['T0_preset']} / {ld['T1_preset']} = {ld['T0_s']} / {ld['T1_s']} s; settings "
            f"{lc.get('plc_t0_s')} / {lc.get('plc_t1_s')} s", OK if same else FAIL)
        missing = [r["id"] for r in ld["requirements"] if r["status"] in ("ABSENT", "MISMATCH")]
        add("Ladder requirements R1-R13", "all PRESENT", "absent / mismatch: " + (", ".join(missing) or "none"),
            OK if not missing else WARN)
    if lc:
        cal = lc.get("speed_calibration")
        spread = (cal or {}).get("spread_pct")
        add("Belt speed", f"measured (calibration record), spread <= {S['speed_spread_pct']} %",
            f"{lc.get('conveyor_mm_s')} mm/s - {lc['speed_source']}"
            + (f", spread {spread} %" if spread is not None else ""),
            NOT if not cal else OK if spread is None or spread <= S["speed_spread_pct"] else WARN)
        add("Inspection -> reject distance", "measured on the machine", f"{lc.get('inspection_to_reject_mm')} mm (SAVED)",
            WARN if lc.get("inspection_to_reject_mm") else NOT)
        add("FAULT action", "REJECT (a fault is never a PASS)", str(lc["fault_action"]),
            OK if str(lc["fault_action"]).upper() == "REJECT" else FAIL)
        add("Live threshold tuning", "none in production (thresholds = the activated model's)",
            f"shift {lc['threshold_shift']}, {len(lc['threshold_overrides'])} overrides",
            OK if not lc["threshold_shift"] and not lc["threshold_overrides"] else WARN)
        add("E-stop status input", "configured (and hardwired E-stop verified separately)",
            str(lc.get("estop_device") or "none"), OK if lc.get("estop_device") else WARN)
    cm = rep.get("cameras")
    if cm is None:
        add("Line cameras connected", f"{S['line_cameras']} line cameras", "not measured (--no-cameras)", NOT)
    else:
        saved = cm.get("line_cameras_saved") or []
        present = [i for i in saved if any(f["index"] == i for f in cm["found"])]
        line_ok = len(saved) >= S["line_cameras"] and len(present) == len(saved)
        add("Line cameras connected", f"{S['line_cameras']} line cameras (settings line_cameras) delivering frames",
            f"saved {saved}, connected {present}; all found: "
            f"{', '.join(str(f['index']) + ' ' + (f['name'] or '?') for f in cm['found']) or 'none'}"
            + ("" if line_ok else "  -> the rows below measure OTHER cameras, not the line"),
            OK if line_ok else FAIL)
        sv = cm.get("survey") or {}
        for i, r in (sv.get("alone") or {}).items():
            fps = r.get("fps") or 0
            add(f"Camera {i} alone", f">= {S['cam_alone_fps']} fps at {S['line_resolution'][0]}x{S['line_resolution'][1]}",
                f"{r.get('size', '?')} {r.get('fourcc', '')} {fps} fps, sharpness {r.get('sharpness', '?')}",
                OK if fps >= S["cam_alone_fps"] and r.get("size") == "%dx%d" % S["line_resolution"] else WARN)
        tog = (sv.get("together") or {})
        key = "%dx%d" % S["line_resolution"]
        if key in tog:
            parts = [f"cam {i}: {v.get('size', '?')} {v.get('fps')} fps" for i, v in tog[key].items()]
            ok = all((v.get("fps") or 0) >= S["cam_together_fps"] and v.get("size") == key for v in tog[key].values())
            alt = ", ".join(f"{res}: " + "/".join(str(v.get("fps")) for v in m.values()) + " fps"
                            for res, m in tog.items() if res != key)
            add("Both cameras together", f">= {S['cam_together_fps']} fps each at {key}", "; ".join(parts)
                + (f"  (also {alt})" if alt else ""), OK if ok else WARN)
    md = rep.get("models")
    if md is None:
        add("Models", "active classifier + detector load", "not measured (--no-models)", NOT)
    else:
        c, d = md["classifier"], md["detector"]
        add("Classifier", f"active checkpoint loads, <= {S['classifier_ms']} ms/frame",
            c.get("error") or f"{c['active_model']} {c.get('arch')} on {c.get('device')}: {c.get('ms_mean')} ms "
                               f"(p95 {c.get('ms_p95')})",
            FAIL if c.get("error") else OK if c["ms_mean"] <= S["classifier_ms"] else WARN)
        add("Detector", f"weights present + provenance, <= {S['detector_ms']} ms/frame",
            d.get("error") or (f"{d['weights']} {d.get('model_id')} on {d.get('device')}: {d.get('ms_mean')} ms "
                               f"(p95 {d.get('ms_p95')})" if d.get("present") else "weights missing"),
            FAIL if d.get("error") or not d.get("present") else OK if d["ms_mean"] <= S["detector_ms"] else WARN)
    return rows


# ------------------------------------------------------------------------------------------------ report
def _md(rep: dict, rows: list) -> str:
    L = [f"# System report - {rep['when']}", "",
         "Measured by `python -m tools.system_report`. **MEASURED** = this run, **SAVED** = from settings / files "
         "(may be stale), **NOT MEASURED** = nobody has measured it. Specification: `SPEC` in tools/system_report.py.",
         "", "## Specification check", "", "| Item | Required | Found | Status |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {str(c).replace('|', '/')} | **{d}** |" for a, b, c, d in rows]
    cnt = {s: sum(1 for r in rows if r[3] == s) for s in (OK, WARN, FAIL, NOT)}
    L += ["", f"**{cnt[OK]} OK, {cnt[WARN]} WARN, {cnt[FAIL]} FAIL, {cnt[NOT]} NOT MEASURED.** "
              "A software report is not a hardware test: the physical cycle is proven only on the machine.", ""]
    lc = rep.get("line") or {}
    if lc.get("travel_s") is not None:
        L += ["## Line timing derived from the SAVED values", "",
              f"- travel inspection -> reject: {lc['inspection_to_reject_mm']} mm / {lc['conveyor_mm_s']} mm/s = "
              f"**{lc['travel_s']} s**",
              f"- REJECT command (M1) sent {lc['reject_dispatch_after_trigger_s']} s after the trigger (travel - T0)",
              f"- one REJECT keeps the PLC from accepting a new trigger for about {lc['reject_cycle_blocks_trigger_s']} s "
              f"(at most {60 / lc['reject_cycle_blocks_trigger_s']:.1f} rejected bottles/min with the one-bottle handshake)",
              ""]
    for key, title in (("computer", "Computer"), ("serial", "Serial ports"), ("plc", "PLC link (read-only)"),
                       ("ladder", "Saved PLC ladder"), ("line", "Line settings (SAVED)"), ("cameras", "Cameras"),
                       ("models", "Models"), ("data", "Data")):
        if rep.get(key) is not None:
            L += [f"## {title}", "", "```json", json.dumps(rep[key], indent=2, default=str)[:12000], "```", ""]
    return "\n".join(L)


def run(argv: list, out_root: Path | None = None) -> tuple:
    from vision import dataset as D
    settings = D.load_settings()
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    folder = (out_root or ROOT / "reports" / "system") / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    rep = {"when": when, "project": D.PROJECT}
    steps = [("computer", lambda: computer()), ("serial", lambda: serial_ports(settings)),
             ("ladder", lambda: ladder(settings)), ("line", lambda: line_config(settings)), ("data", data)]
    if "--plc" in argv:
        port = argv[argv.index("--port") + 1] if "--port" in argv else None
        steps.append(("plc", lambda: plc_link(settings, port)))
    if "--no-cameras" not in argv:
        steps.append(("cameras", lambda: cameras(settings, folder, 3.0)))
    if "--no-models" not in argv:
        steps.append(("models", lambda: models(settings, 20)))
    for key, fn in steps:
        print(f"-- {key} ...", flush=True)
        try:
            rep[key] = fn()
        except Exception as e:                                       # noqa: BLE001 - one section never stops the rest
            rep[key] = {"error": f"{type(e).__name__}: {e}"}
    rows = judge(rep)
    rep["spec_check"] = [dict(zip(("item", "required", "found", "status"), r)) for r in rows]
    (folder / "system_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (folder / "system_report.md").write_text(_md(rep, rows), encoding="utf-8")
    return folder, rows


def selftest() -> None:
    from vision import dataset as D
    before = D.SETTINGS_JSON.read_bytes() if D.SETTINGS_JSON.exists() else None
    with tempfile.TemporaryDirectory() as tmp:
        folder, rows = run(["--no-cameras", "--no-models"], Path(tmp))
        rep = json.loads((folder / "system_report.json").read_text(encoding="utf-8"))
        assert (folder / "system_report.md").is_file()
        assert "plc" not in rep, "the PLC must not be touched without --plc"
        items = {r[0]: r[3] for r in rows}
        assert items["Line cameras connected"] == NOT and items["Models"] == NOT and items["PLC link (read-only)"] == NOT, items
        assert items["FAULT action"] in (OK, FAIL) and rep["computer"]["python"]
    # judgement on synthetic values: each rule says what it should
    fake = {"line": {**line_config({"conveyor_mm_s": 100, "inspection_to_reject_mm": 1000, "plc_t0_s": 1, "plc_t1_s": 0.5,
                                    "fault_action": "PASS"})},
            "ladder": {"T0_s": 1.0, "T1_s": 0.5, "T0_preset": "K10", "T1_preset": "K5", "requirements": []},
            "plc": {"port": "COM9", "error": "SerialException: Access is denied"}}
    got = {r[0]: r[3] for r in judge(fake)}
    assert got["FAULT action"] == FAIL and got["T0 / T1 (saved ladder vs settings)"] == OK, got
    assert got["Belt speed"] == NOT and got["PLC link (read-only)"] == FAIL, got
    assert fake["line"]["travel_s"] == 10.0 and fake["line"]["reject_cycle_blocks_trigger_s"] == 10.5
    after = D.SETTINGS_JSON.read_bytes() if D.SETTINGS_JSON.exists() else None
    assert before == after, "settings.json was modified"
    print(f"ok  system report: {len(rows)} spec rows, no device opened, settings untouched")


def main() -> None:
    if "--selftest" in sys.argv:
        return selftest()
    t = time.perf_counter()
    folder, rows = run(sys.argv[1:])
    print()
    w = max(len(r[0]) for r in rows)
    for item, req, found, status in rows:
        print(f"  {status:<12} {item:<{w}}  {str(found)[:110]}")
    print(f"\nReport: {folder / 'system_report.md'}   ({time.perf_counter() - t:.0f} s)")


if __name__ == "__main__":
    main()
