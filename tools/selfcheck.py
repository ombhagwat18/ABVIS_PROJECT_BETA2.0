"""One command / one button for "is the CODE correct?": runs every module self-test in its own process.

    python selfcheck.py            # quick set (no window, no GPU, ~1-2 min)
    python selfcheck.py --full     # + the model/data checks and the GUI self-test (builds the real window, minutes)

Each test is a software check with fakes (fake PLC emulating the decoded ladder, fake cameras, fake models).
A pass is NOT a hardware test. A failure prints the last lines of that test's output.
The ladder is checked separately: python -m plc.ladder_sim (docs/hardware/PLC_LADDER_REQUIREMENTS.md).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, argv after python, what it proves)
QUICK = (
    ("Decision engine", ["-m", "line.decision"], "recipe rules, frame vote, camera fusion, every failure -> FAULT"),
    ("Machine cycle", ["-m", "line.machine_cycle"], "trigger -> inspection -> FIFO -> M0/M1 against a fake PLC running the decoded ladder; "
                                            "staggered cameras, association fault, late reject, halt latch, record"),
    ("Time tracking", ["-m", "line.tracking"], "camera stations, association window, speed calibration"),
    ("Machine state", ["-m", "line.machine_state"], "the one machine state + start checklist"),
    ("Alarms", ["-m", "line.alarms"], "coded alarms, acknowledge, dedup"),
    ("Production record", ["-m", "line.production_store"], "SQLite runs / bottles / alarms, evidence, shifts"),
    ("Event logs", ["-m", "line.applog"], "per-subsystem log files and search"),
    ("Model registry", ["-m", "registry.model_registry", "--selftest"], "validate / approve / activate gates, rollback"),
    ("PLC protocol", ["-m", "plc.test_simulation"], "Modbus frames, addressing, write policy, faults"),
    ("PLC service", ["-m", "plc.test_service"], "trigger handling, one answer per trigger, no retries"),
    ("PLC commissioning window", ["-m", "plc.commissioning", "--selftest"], "armed test commands"),
    ("Ladder decoder", ["-m", "plc.ladder_check", "--selftest"], ".isp reader and requirement checks"),
    ("Ladder simulator", ["-m", "plc.ladder_sim", "--selftest"], "scan simulator and its scenarios"),
    ("Active learning", ["-m", "vision.autoannotate"], "proposals kept out of labels, review queues"),
    ("Stable verdict", ["-m", "vision.verdict"], "one GOOD / DEFECT per bottle: settle, vote, latch until the bottle leaves, FAULT never GOOD"),
    ("Production export", ["-m", "line.production_export"], "plain-word rows, CSV for Excel, printable report"),
    ("Threshold calibration", ["-m", "vision.calibrate_thresholds", "--selftest"], "validation-chosen thresholds, recall guard"),
    ("Model activation gates", ["-m", "registry.model_checks", "--selftest"], "real-camera limits, background-shortcut check"),
)
FULL = (
    ("Dataset / split", ["-m", "vision.dataset"], "labels, crop pipeline, scene split"),
    ("Training maths", ["-m", "vision.train", "--demo"], "metrics"),
    ("Inference", ["-m", "vision.infer"], "PASS / REJECT / FAULT freshness rules"),
    ("Detector runtime", ["-m", "vision.detect"], "detector contract + real-checkpoint sanity check"),
    ("Segmentation interface", ["-m", "vision.segment"], "fake model"),
    ("Stage 2 split v3", ["stage2_dataset/split_v3.py", "--selftest"], "missing-cap scene in train"),
    ("GUI (all pages, fake PLC, fake cameras)", ["-m", "ui.app", "--selftest"], "builds the real window and runs the line"),
)


def run_one(argv: list, timeout: int = 900) -> tuple:
    """(ok, last line, seconds)."""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable] + argv, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s", time.time() - t0
    lines = [x for x in (p.stdout + "\n" + p.stderr).splitlines()
             if x.strip() and "torch.load" not in x and "FutureWarning" not in x and not x.startswith("  File")]
    ok = p.returncode == 0
    return ok, ("\n".join(lines[-6:]) if not ok else (lines[-1] if lines else "ok"))[:600], time.time() - t0


def run_all(full: bool = False, progress=None) -> list:
    """[(label, ok, message, seconds, what it proves)]; progress(label, result) after each test."""
    out = []
    for label, argv, why in QUICK + (FULL if full else ()):
        ok, msg, sec = run_one(argv)
        out.append((label, ok, msg, sec, why))
        if progress:
            progress(label, out[-1])
    return out


def main() -> None:
    res = run_all("--full" in sys.argv, lambda l, r: print(f"{'ok  ' if r[1] else 'FAIL'} {l:<42} {r[3]:5.1f}s"))
    bad = [r for r in res if not r[1]]
    for r in bad:
        print(f"\n--- {r[0]} ---\n{r[2]}")
    print(f"\n{len(res) - len(bad)}/{len(res)} passed" + ("" if not bad else f", {len(bad)} FAILED"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
