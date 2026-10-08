"""Camera placement calculator (small Tk tool; the maths are plain functions, self-tested).

    python camera_planner.py            # open the tool
    python camera_planner.py --selftest

Inputs are what you MEASURE on the machine (distances, belt speed, the camera's real field of view). The default FOV
(65 x 40 degrees landscape) is a typical 4K webcam value, NOT a measured EMEET Nova 4K figure: measure it by
standing a ruler at a known distance and reading how wide / tall the frame is (fov = 2*atan(width / 2 / distance)).
The output is what to put in settings.json (capture_delay_s, inspect_window_s) and where the ROI will fall.
Nothing here talks to a camera or the PLC and nothing is written to settings.json.
"""
from __future__ import annotations

import math
import sys

ENCLOSURE_MM = (450.0, 320.0, 400.0)         # L x W x H, stated by the project owner


def coverage_mm(distance_mm: float, fov_deg: float) -> float:
    """Width of the scene a camera sees at `distance_mm` for a field of view of `fov_deg`."""
    return 2.0 * distance_mm * math.tan(math.radians(fov_deg) / 2.0)


def min_distance_mm(span_mm: float, fov_deg: float, margin: float = 0.2) -> float:
    """Closest camera distance at which `span_mm` (+ margin) still fits in the field of view."""
    return span_mm * (1.0 + margin) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def plan(distance_mm, fov_long_deg, fov_short_deg, res_long_px, res_short_px, bottle_w_mm, bottle_h_mm,
         speed_mm_s, sensor_to_cam_mm, exposure_s=1 / 250, window_s=1.5, portrait=True, margin=0.2) -> dict:
    """Everything derived from the measured numbers. portrait=True: the camera's long side is vertical."""
    fov_v, fov_h = (fov_long_deg, fov_short_deg) if portrait else (fov_short_deg, fov_long_deg)
    res_v, res_h = (res_long_px, res_short_px) if portrait else (res_short_px, res_long_px)
    cov_h, cov_v = coverage_mm(distance_mm, fov_h), coverage_mm(distance_mm, fov_v)
    ppm = res_h / cov_h                                         # pixels per mm at the bottle
    fits = cov_v >= bottle_h_mm * (1 + margin) and cov_h >= bottle_w_mm * (1 + margin)
    in_view_s = (cov_h + bottle_w_mm) / speed_mm_s              # bottle enters -> leaves the frame
    arrive_s = sensor_to_cam_mm / speed_mm_s                    # photo-eye -> bottle at the optical axis
    return {
        "coverage_w_mm": cov_h, "coverage_h_mm": cov_v, "px_per_mm": ppm,
        "bottle_px": (bottle_w_mm * ppm, bottle_h_mm * ppm),
        "fits": fits,
        "min_distance_mm": max(min_distance_mm(bottle_h_mm, fov_v, margin), min_distance_mm(bottle_w_mm, fov_h, margin)),
        "blur_px": speed_mm_s * exposure_s * ppm,               # smear of a moving edge during one exposure
        "arrive_s": arrive_s,
        "in_view_s": in_view_s,
        "capture_delay_s": max(0.0, arrive_s - window_s / 2.0),
        "window_max_s": min(in_view_s, 2.0 * arrive_s),
        "roi_px": (round(bottle_w_mm * ppm * (1 + margin)), round(bottle_h_mm * ppm * (1 + margin))),
        "wall_ok": distance_mm <= ENCLOSURE_MM[1] / 2.0,        # camera must sit inside the 320 mm width
    }


def report(p: dict) -> str:
    yn = lambda ok: "OK" if ok else "NO"                                        # noqa: E731
    return "\n".join([
        f"scene at the bottle      {p['coverage_w_mm']:.0f} x {p['coverage_h_mm']:.0f} mm   ({p['px_per_mm']:.1f} px/mm)",
        f"bottle in the image      {p['bottle_px'][0]:.0f} x {p['bottle_px'][1]:.0f} px   fits with margin: {yn(p['fits'])}",
        f"closest usable distance  {p['min_distance_mm']:.0f} mm  (camera inside the 320 mm width: {yn(p['wall_ok'])})",
        f"motion blur              {p['blur_px']:.1f} px per exposure  ({yn(p['blur_px'] < 2)}: want < 2 px)",
        f"photo-eye -> optical axis {p['arrive_s']:.2f} s",
        f"bottle visible for       {p['in_view_s']:.2f} s   -> inspect_window_s must be <= {p['window_max_s']:.2f}",
        "",
        "settings.json:",
        f'   "capture_delay_s": {p["capture_delay_s"]:.2f}',
        f"ROI (crop around bottle)  {p['roi_px'][0]} x {p['roi_px'][1]} px; run calibrate.py after mounting.",
    ])


def selftest():
    assert abs(coverage_mm(100, 90) - 200) < 1e-6
    p = plan(distance_mm=160, fov_long_deg=65, fov_short_deg=40, res_long_px=1920, res_short_px=1080,
             bottle_w_mm=60, bottle_h_mm=165, speed_mm_s=100, sensor_to_cam_mm=250)
    assert p["fits"] and abs(p["arrive_s"] - 2.5) < 1e-9 and abs(p["capture_delay_s"] - 1.75) < 1e-9, p
    assert not plan(distance_mm=60, fov_long_deg=65, fov_short_deg=40, res_long_px=1920, res_short_px=1080,
                    bottle_w_mm=60, bottle_h_mm=165, speed_mm_s=100, sensor_to_cam_mm=250)["fits"]
    assert abs(min_distance_mm(100, 90, 0.0) - 50) < 1e-9
    print("ok  camera planner: coverage, fit check, arrival time -> capture_delay_s, blur, ROI")


FIELDS = (("distance_mm", "Camera to bottle centre (mm)", 160), ("fov_long_deg", "Camera long-side FOV (deg) MEASURE", 65),
          ("fov_short_deg", "Camera short-side FOV (deg) MEASURE", 40), ("res_long_px", "Resolution long side (px)", 1920),
          ("res_short_px", "Resolution short side (px)", 1080), ("bottle_w_mm", "Bottle width (mm)", 60),
          ("bottle_h_mm", "Bottle height (mm)", 165), ("speed_mm_s", "Belt speed (mm/s) MEASURE", 100),
          ("sensor_to_cam_mm", "Photo-eye to camera axis (mm) MEASURE", 250), ("exposure_s", "Exposure (s)", 0.004),
          ("window_s", "Inspect window (s)", 1.5))


def main():
    import tkinter as tk
    root = tk.Tk()
    root.title("Camera placement calculator")
    ent = {}
    left = tk.Frame(root)
    left.pack(side="left", padx=10, pady=10, anchor="n")
    for i, (k, label, dflt) in enumerate(FIELDS):
        tk.Label(left, text=label, anchor="w").grid(row=i, column=0, sticky="w")
        e = tk.Entry(left, width=9)
        e.insert(0, str(dflt))
        e.grid(row=i, column=1, padx=6, pady=1)
        ent[k] = e
    portrait = tk.BooleanVar(value=True)
    tk.Checkbutton(left, text="portrait mounting (rotate 90)", variable=portrait).grid(row=len(FIELDS), columnspan=2, sticky="w")
    out = tk.Label(root, justify="left", font=("Consolas", 10), anchor="nw", width=64)
    out.pack(side="left", padx=10, pady=10, anchor="n")
    cv = tk.Canvas(root, width=360, height=260, bg="white")
    cv.pack(side="left", padx=10, pady=10, anchor="n")

    def go(*_):
        try:
            v = {k: float(e.get()) for k, e in ent.items()}
            p = plan(**v, portrait=portrait.get())
        except (ValueError, ZeroDivisionError) as e:
            out.configure(text=f"fix the inputs: {e}")
            return
        out.configure(text=report(p))
        cv.delete("all")                                  # top view, enclosure 450 x 320 mm at 0.7 px/mm
        k, L, W = 0.7, ENCLOSURE_MM[0], ENCLOSURE_MM[1]
        x0, y0 = 20, 20
        cv.create_rectangle(x0, y0, x0 + L * k, y0 + W * k)
        by = y0 + W * k / 2
        cv.create_line(x0, by, x0 + L * k, by, width=10, fill="#4a4")
        cv.create_text(x0 + 4, by - 14, anchor="w", text="belt ->")
        cx = x0 + L * k / 2
        d = v["distance_mm"] * k
        fov_h = v["fov_short_deg"] if portrait.get() else v["fov_long_deg"]
        half = math.tan(math.radians(fov_h) / 2) * v["distance_mm"] * k
        cv.create_polygon(cx, by - d, cx - half, by, cx + half, by, fill="#cde", outline="#58a")
        cv.create_rectangle(cx - 6, by - d - 6, cx + 6, by - d + 6, fill="black")
        cv.create_oval(cx - v["bottle_w_mm"] * k / 2, by - v["bottle_w_mm"] * k / 2,
                       cx + v["bottle_w_mm"] * k / 2, by + v["bottle_w_mm"] * k / 2, fill="#fc6")
        cv.create_text(cx, y0 + W * k + 14, text="top view: camera (black), view cone, bottle")
    for e in ent.values():
        e.bind("<KeyRelease>", go)
    portrait.trace_add("write", go)
    go()
    root.mainloop()


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
