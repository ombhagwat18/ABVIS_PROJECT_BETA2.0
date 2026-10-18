"""The one clock every timestamp on the inspection path is taken from.

Frame stamps (vision/infer.py), PLC triggers and events (plc/service.py), bottle deadlines (line/machine_cycle.py) and
tracking windows are compared with each other, so they must come from the SAME clock. That clock is
time.perf_counter: monotonic, never adjusted by NTP or the user, and high resolution.

Why not time.monotonic: on Python 3.8 / Windows it is GetTickCount64, which ticks every 15.625 ms (measured on the
line PC, 2026-10-13: time.get_clock_info('monotonic').resolution == 0.015625). Adjacent camera frames (33 ms apart)
then share stamps and a frame up to one tick before a trigger can look as if it came after it. perf_counter is
QueryPerformanceCounter there (100 ns). Wall-clock time (time.time) is for display and records only, never for
ordering or deadlines.

    from line import clock
    t = clock.now()
"""
from __future__ import annotations

import time

now = time.perf_counter


def resolution() -> float:
    """Seconds per tick of now(), as the OS reports it."""
    return time.get_clock_info("perf_counter").resolution


def demo():
    info = time.get_clock_info("perf_counter")
    assert info.monotonic and not info.adjustable, info
    assert resolution() <= 1e-3, f"clock too coarse for frame ordering: {resolution()} s"
    a = now()
    stamps = [now() for _ in range(1000)]
    assert all(b >= a for b in stamps) and stamps == sorted(stamps), "clock went backwards"
    # 1000 back-to-back reads must not all share one tick (GetTickCount64 would give ~1 distinct value)
    assert len(set(stamps)) > 100, f"only {len(set(stamps))} distinct stamps: resolution too coarse"
    print(f"ok  (clock = perf_counter, {info.implementation}, resolution {info.resolution:.1e} s, "
          f"{len(set(stamps))}/1000 distinct back-to-back reads)")


if __name__ == "__main__":
    demo()
