"""Structured, event-based log files, one per subsystem: app, camera, ai, plc, machine, alarm, production.

    logs/<channel>.log      rotating (2 MB x 5), one line per EVENT:
    2026-10-05 17:40:12.345 | WARNING | camera | CAMERA_STALE slot=Camera 0 | last frame 1.4 s ago

Events, never frames: a camera logs open / fault / reconnect, not every image; the line logs one line per
finished bottle. Nothing here decides anything; a failure to log never stops the machine.
search() reads the files back for the Health page (filter by channel, level, text).

    python applog.py      # self-test (temporary folder)
"""
from __future__ import annotations

import logging
import logging.handlers
import threading
from pathlib import Path

CHANNELS = ("app", "camera", "ai", "plc", "machine", "alarm", "production")
ROOT = Path(__file__).resolve().parent / "logs"
_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(channel)s | %(message)s"
_lock = threading.Lock()
_folder: Path | None = None
_loggers: dict = {}


def setup(folder: Path | None = None) -> Path:
    """(Re)point every channel at `folder` (default ./logs). Safe to call more than once."""
    global _folder
    with _lock:
        _folder = Path(folder) if folder else ROOT
        _folder.mkdir(parents=True, exist_ok=True)
        for ch in CHANNELS:
            lg = logging.getLogger(f"inspection.{ch}")
            lg.setLevel(logging.INFO)
            lg.propagate = False
            for h in list(lg.handlers):
                lg.removeHandler(h)
                h.close()
            h = logging.handlers.RotatingFileHandler(str(_folder / f"{ch}.log"), maxBytes=2_000_000, backupCount=5,
                                                     encoding="utf-8", delay=True)
            h.setFormatter(logging.Formatter(_FMT, "%Y-%m-%d %H:%M:%S"))
            lg.addHandler(h)
            _loggers[ch] = logging.LoggerAdapter(lg, {"channel": ch})
    return _folder


def log(channel: str, message: str, level: str = "INFO", **fields):
    """One event line. fields are appended as key=value. Never raises."""
    try:
        if _folder is None:
            setup()
        lg = _loggers.get(channel) or _loggers["app"]
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        lg.log(getattr(logging, level.upper(), logging.INFO), f"{message}{' | ' + extra if extra else ''}")
    except Exception:                                                # noqa: BLE001 - logging must never stop the line
        pass


def folder() -> Path:
    return _folder or ROOT


def search(channel: str | None = None, text: str = "", level: str | None = None, limit: int = 300) -> list:
    """Newest-first matching lines from the current log files (not the rotated backups)."""
    out = []
    for ch in ([channel] if channel else CHANNELS):
        p = folder() / f"{ch}.log"
        if not p.exists():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            if text and text.lower() not in ln.lower():
                continue
            if level and f"| {level.upper():<7} |" not in ln:
                continue
            out.append(ln)
    out.sort(reverse=True)                                           # lines start with the timestamp
    return out[:limit]


def alarm_listener(a):
    """alarms.AlarmManager on_change hook -> alarm.log."""
    lvl = {"CRITICAL": "CRITICAL", "MAJOR": "ERROR", "WARNING": "WARNING"}.get(a.severity, "INFO")
    log("alarm", f"{a.code} {a.state}", lvl if a.state == "ACTIVE" else "INFO", key=a.key or "-", count=a.count,
        cause=(a.cause or "")[:200])


def plc_listener(ev):
    """PLCService listener -> plc.log (events only; the poll itself is not logged)."""
    if ev.event in ("STOPPED",):
        return
    lvl = "WARNING" if ev.event in ("FAULT", "CONNECT_FAILED", "TRIGGER_LOST", "BOTTLE_UNTRIGGERED") \
        or "FAILED" in ev.event or "NOT_ACKED" in ev.event or "LOST" in ev.event else "INFO"
    log("plc", ev.event, lvl, device=ev.device or "-", value=ev.ack or "-", trigger=ev.trigger_id,
        ms=None if ev.latency_ms is None else round(ev.latency_ms, 1), error=(ev.error or "-")[:160])


def demo():
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as td:
        setup(Path(td))
        log("camera", "opened", source=2, wh="1920x1080")
        log("plc", "FAULT", "WARNING", error="COM5 access denied")
        log("production", "bottle", final="REJECT", id="000007")

        class A:
            code, state, severity, key, count, cause = "E_STOP", "ACTIVE", "CRITICAL", "", 1, "X3 open"
        alarm_listener(A())
        time.sleep(0.01)
        assert search("camera")[0].endswith("opened | source=2 wh=1920x1080"), search("camera")
        assert len(search(text="com5")) == 1 and search(level="CRITICAL")[0].split(" | ")[2] == "alarm"
        assert len(search()) == 4 and all(" | " in s for s in search())
        for lg in _loggers.values():
            for h in lg.logger.handlers:
                h.close()
    setup(ROOT)
    print("ok  applog: 7 channels, one event per line, search by channel / text / level, alarm + plc hooks")


if __name__ == "__main__":
    demo()
