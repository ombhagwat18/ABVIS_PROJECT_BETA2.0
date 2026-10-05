"""Where a bottle is on the belt, from TIME (this machine has no encoder).

    trigger (X0 photo-eye -> M2) at t0
        -> camera station k, offset_mm downstream of the photo-eye: expected at t0 + offset_mm / speed
        -> reject cylinder, inspection_to_reject_mm downstream:      expected at t0 + travel

Every distance and the belt speed are MEASURED values from settings.json; nothing here has a default
distance or speed (0 = "not measured"). Time tracking is NOT encoder tracking: a belt that slips or
changes speed moves every expected time, so the calibrated speed spread (speed_tolerance_pct) is
turned into an explicit timing uncertainty and checked before the line may start.

PositionSource is the seam a future encoder plugs into: MachineCycle asks "when will the bottle that
triggered at t be D mm further on?" and never cares how that is answered.

    TimePositionSource      the only implementation in use (measured speed, constant-speed model)
    EncoderPositionSource   placeholder: refuses to be built until an encoder exists

Pure functions + small classes, no I/O.   python tracking.py   # self-test
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field

TRACKING_DEFAULTS = {
    "camera_stations": {},          # {"<camera source>": {"name", "role", "offset_mm", "side", "rules": {...}}}
    "speed_tolerance_pct": 10.0,    # largest accepted calibrated speed spread ((max-min)/mean, %)
    "timing_margin_s": 0.15,        # slack required between "decision ready" and "REJECT must be sent"
    "speed_calibration": None,      # last calibrate_speed() record that was saved (timestamped)
}

ROLES = ("general", "body / label", "cap / neck", "opposite label", "redundant")


# --------------------------------------------------------------------------------- position sources
class PositionSource:
    """How long the belt needs to carry a bottle `distance_mm` (and how sure we are)."""
    kind = "abstract"

    @property
    def measured(self) -> bool:
        raise NotImplementedError

    def eta_s(self, distance_mm: float) -> float:
        """Seconds for the belt to carry a bottle distance_mm."""
        raise NotImplementedError

    def uncertainty_s(self, distance_mm: float) -> float:
        """+- seconds on eta_s(distance_mm)."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind


class TimePositionSource(PositionSource):
    """Constant belt speed, measured by the calibration wizard (or typed in). speed 0 = not measured."""
    kind = "time"

    def __init__(self, speed_mm_s: float, tolerance_pct: float = 10.0):
        self.speed = float(speed_mm_s or 0.0)
        self.tolerance = max(0.0, float(tolerance_pct or 0.0)) / 100.0

    @property
    def measured(self) -> bool:
        return self.speed > 0

    def eta_s(self, distance_mm: float) -> float:
        if not self.measured:
            raise ValueError("conveyor speed not measured")
        return float(distance_mm) / self.speed

    def uncertainty_s(self, distance_mm: float) -> float:
        return self.eta_s(distance_mm) * self.tolerance if self.measured else math.inf

    def describe(self) -> str:
        if not self.measured:
            return "time-based, speed NOT measured"
        return f"time-based, {self.speed:.1f} mm/s +-{self.tolerance * 100:.0f}% (no encoder)"


class EncoderPositionSource(PositionSource):
    """Future: counts from a belt encoder. Not built on this machine -- refusing beats pretending."""
    kind = "encoder"

    def __init__(self, *_, **__):
        raise NotImplementedError("no encoder is installed on this machine; use TimePositionSource")


def position_source(cfg: dict) -> PositionSource:
    return TimePositionSource(float(cfg.get("conveyor_mm_s") or 0.0),
                              float(cfg.get("speed_tolerance_pct", TRACKING_DEFAULTS["speed_tolerance_pct"])))


# --------------------------------------------------------------------------------- camera stations
@dataclass
class Station:
    camera: str                     # camera id (str(source)), the key Inspector / decision use
    name: str = ""
    role: str = "general"
    offset_mm: float = 0.0          # downstream of the trigger photo-eye (0 = at the trigger point)
    side: str = ""                  # e.g. "left wall" / "right wall": documentation only
    rules: dict = field(default_factory=dict)   # per-camera decision overrides (station_x, judge, ...)


def stations(cfg: dict, cameras) -> dict:
    """{camera id: Station} for the line cameras; unconfigured cameras sit at the trigger point."""
    raw = cfg.get("camera_stations") or {}
    out = {}
    for i, cam in enumerate(cameras):
        cid = str(cam)
        s = raw.get(cid) or {}
        out[cid] = Station(cid, str(s.get("name") or f"Camera {i + 1}"), str(s.get("role") or "general"),
                           float(s.get("offset_mm") or 0.0), str(s.get("side") or ""), dict(s.get("rules") or {}))
    return out


def station_offsets_s(cfg: dict, cameras) -> dict:
    """{camera id: seconds after the trigger the bottle is in front of it}. Raises ValueError when a
    camera is downstream but the belt speed is not measured (its window cannot be placed)."""
    src = position_source(cfg)
    out = {}
    for cid, st in stations(cfg, cameras).items():
        if st.offset_mm <= 0:
            out[cid] = 0.0
        elif not src.measured:
            raise ValueError(f"{st.name} is {st.offset_mm:g} mm after the trigger but the conveyor speed is not "
                             f"measured: run the speed calibration first")
        else:
            out[cid] = src.eta_s(st.offset_mm)
    return out


def station_problems(cfg: dict, cameras) -> list:
    """Why the camera-station layout cannot be tracked by time ([] = usable). The line refuses to start."""
    out = []
    src = position_source(cfg)
    st = stations(cfg, cameras)
    dist = float(cfg.get("inspection_to_reject_mm") or 0.0)
    window = float(cfg.get("inspect_window_s") or 0.6)
    delay = float(cfg.get("capture_delay_s") or 0.0)
    margin = float(cfg.get("timing_margin_s", TRACKING_DEFAULTS["timing_margin_s"]))
    t0 = float(cfg.get("plc_t0_s") or 0.0)
    for s in st.values():
        if s.offset_mm < 0:
            out.append(f"{s.name}: offset {s.offset_mm:g} mm is upstream of the trigger sensor (not supported)")
    downstream = [s for s in st.values() if s.offset_mm > 0]
    if not downstream:
        return out
    if not src.measured:
        out.append("a camera is downstream of the trigger but the conveyor speed is not measured")
        return out
    if dist <= 0:
        out.append("a camera is downstream of the trigger but the inspection->reject distance is not measured")
        return out
    for s in downstream:
        if s.offset_mm >= dist:
            out.append(f"{s.name} ({s.offset_mm:g} mm) is at or after the reject station ({dist:g} mm)")
    far = max(downstream, key=lambda s: s.offset_mm)
    ready = src.eta_s(far.offset_mm) + src.uncertainty_s(far.offset_mm) + delay + window
    dispatch = src.eta_s(dist) - t0
    if ready + margin > dispatch:
        out.append(f"{far.name} sees the bottle too late: decision ready ~{ready:.2f} s after the trigger "
                   f"(+{margin:.2f} s margin) but a REJECT must be sent {dispatch:.2f} s after it "
                   f"(travel {src.eta_s(dist):.2f} s - T0 {t0:.2f} s)")
    return out


# --------------------------------------------------------------------------------- association
def frame_window(t_trigger: float, offset_s: float, window_s: float, delay_s: float = 0.0) -> tuple:
    """(start, end) monotonic window in which a camera `offset_s` downstream sees this bottle."""
    start = t_trigger + offset_s + delay_s
    return start, start + window_s


def association_problem(t_trigger: float, offset_s: float, window_s: float, uncertainty_s: float,
                        sensed) -> str | None:
    """CAMERA_ASSOCIATION_FAULT check for a downstream camera.

    sensed: monotonic times other bottles were SENSED at the trigger point (triggered or not). A
    neighbour closer in time than the camera window + twice the arrival uncertainty can be in that
    camera's view while this bottle's frames are taken, so they cannot be told apart by time: the
    camera's evidence is refused instead of being given to the wrong bottle. At the trigger point
    (offset 0) the trigger itself fixes the bottle and this check does not apply."""
    if offset_s <= 0:
        return None
    guard = window_s + 2 * uncertainty_s
    near = [t for t in sensed if t != t_trigger and abs(t - t_trigger) < guard]
    if near:
        gap = min(abs(t - t_trigger) for t in near)
        return (f"another bottle was sensed {gap * 1000:.0f} ms from this one; at the downstream camera the two "
                f"cannot be separated by time (window {window_s:.2f} s + 2 x {uncertainty_s:.2f} s uncertainty)")
    return None


# --------------------------------------------------------------------------------- calibration
def calibrate_speed(distance_mm: float, times_s, tolerance_pct: float = 10.0) -> dict:
    """Belt speed from repeated timings of one bottle over a measured distance.

    Returns a record (saved as settings "speed_calibration"): speeds, mean/min/max, spread %, and
    `problem` when the run cannot be trusted. Nothing is invented: too few runs or a bad distance is a
    problem, not a default."""
    rec = {"distance_mm": float(distance_mm), "times_s": [round(float(t), 4) for t in times_s],
           "tolerance_pct": float(tolerance_pct), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "method": "time-based (stopwatch / photo-eye timing), no encoder"}
    good = [float(t) for t in times_s if float(t) > 0]
    if float(distance_mm) <= 0:
        rec["problem"] = "distance must be measured (> 0 mm)"
        return rec
    if len(good) < 3:
        rec["problem"] = f"need at least 3 timed runs (have {len(good)})"
        return rec
    speeds = [float(distance_mm) / t for t in good]
    mean = statistics.mean(speeds)
    spread = (max(speeds) - min(speeds)) / mean * 100
    rec.update(speeds_mm_s=[round(v, 2) for v in speeds], mean_mm_s=round(mean, 2), min_mm_s=round(min(speeds), 2),
               max_mm_s=round(max(speeds), 2), stdev_mm_s=round(statistics.pstdev(speeds), 3),
               spread_pct=round(spread, 2), mean_travel_s=round(statistics.mean(good), 4))
    if spread > float(tolerance_pct):
        rec["problem"] = (f"speed varies {spread:.1f}% between runs (limit {tolerance_pct:g}%): time-based tracking "
                          f"would mis-time the reject. Check belt slip / load, then repeat")
    return rec


def calibration_problem(cfg: dict) -> str | None:
    """Problem with the SAVED speed calibration, if one is saved and its speed is the one in use."""
    rec = cfg.get("speed_calibration")
    if not rec or not cfg.get("conveyor_mm_s"):
        return None
    if rec.get("problem"):
        return f"saved speed calibration is not valid: {rec['problem']}"
    if rec.get("mean_mm_s") and abs(float(rec["mean_mm_s"]) - float(cfg["conveyor_mm_s"])) > 0.5:
        return None                                  # speed typed in after calibrating: the operator's value wins
    if float(rec.get("spread_pct", 0)) > float(cfg.get("speed_tolerance_pct", 10.0)):
        return f"calibrated speed spread {rec['spread_pct']}% exceeds the {cfg.get('speed_tolerance_pct')}% tolerance"
    return None


# --------------------------------------------------------------------------------- self-test
def demo():
    src = TimePositionSource(100.0, 10)
    assert src.measured and abs(src.eta_s(250) - 2.5) < 1e-9 and abs(src.uncertainty_s(250) - 0.25) < 1e-9
    assert not TimePositionSource(0).measured
    try:
        TimePositionSource(0).eta_s(10)
        raise AssertionError("eta with no speed")
    except ValueError:
        pass
    try:
        EncoderPositionSource()
        raise AssertionError("encoder source must refuse to exist")
    except NotImplementedError:
        pass

    base = {"conveyor_mm_s": 100.0, "inspection_to_reject_mm": 600.0, "plc_t0_s": 1.5, "inspect_window_s": 0.4,
            "capture_delay_s": 0.0, "speed_tolerance_pct": 10, "timing_margin_s": 0.15}
    # stations: cam "2" at the trigger point, cam "3" 150 mm downstream on the opposite wall
    cfg = dict(base, camera_stations={"3": {"name": "Camera 2", "role": "cap / neck", "offset_mm": 150,
                                            "side": "right wall", "rules": {"station_x": 0.4}}})
    st = stations(cfg, [2, 3])
    assert st["2"].offset_mm == 0 and st["2"].name == "Camera 1" and st["3"].rules == {"station_x": 0.4}
    off = station_offsets_s(cfg, [2, 3])
    assert off["2"] == 0.0 and abs(off["3"] - 1.5) < 1e-9
    assert station_problems(cfg, [2, 3]) == [], station_problems(cfg, [2, 3])
    # unmeasured speed with a downstream camera: refused, never guessed
    try:
        station_offsets_s(dict(cfg, conveyor_mm_s=0), [2, 3])
        raise AssertionError("offset without speed")
    except ValueError:
        pass
    assert station_problems(dict(cfg, conveyor_mm_s=0), [2, 3])
    # camera 2 too close to the reject station: decision would come after the REJECT dispatch time
    p = station_problems(dict(cfg, camera_stations={"3": {"offset_mm": 400}}), [2, 3])
    assert p and "too late" in p[0], p
    assert "after the reject" in station_problems(dict(cfg, camera_stations={"3": {"offset_mm": 700}}), [2, 3])[0]
    # simultaneous layout (no stations): nothing to check
    assert station_problems(base, [2, 3]) == []

    # association
    assert frame_window(10.0, 1.5, 0.4) == (11.5, 11.9)
    assert association_problem(10.0, 0.0, 0.4, 0.15, [10.1]) is None          # at the trigger: the trigger decides
    assert association_problem(10.0, 1.5, 0.4, 0.15, [10.0, 13.0]) is None     # neighbours far apart
    msg = association_problem(10.0, 1.5, 0.4, 0.15, [10.0, 10.5])
    assert msg and "cannot be separated" in msg, msg

    # calibration
    rec = calibrate_speed(500, [5.0, 5.1, 4.95], 10)
    assert "problem" not in rec and 98 < rec["mean_mm_s"] < 101 and rec["spread_pct"] < 4, rec
    assert "at least 3" in calibrate_speed(500, [5.0, 5.0])["problem"]
    assert "measured" in calibrate_speed(0, [1, 1, 1])["problem"]
    assert "varies" in calibrate_speed(500, [5.0, 6.5, 5.0], 10)["problem"]
    assert calibration_problem({"conveyor_mm_s": 99.5, "speed_calibration": rec}) is None
    bad = calibrate_speed(500, [5.0, 6.5, 5.0], 10)
    assert calibration_problem({"conveyor_mm_s": 90, "speed_calibration": bad})
    print("ok  tracking: time position source (no encoder), encoder placeholder refuses, camera stations / "
          "offsets, layout problems, CAMERA_ASSOCIATION check, speed calibration statistics")


if __name__ == "__main__":
    demo()
