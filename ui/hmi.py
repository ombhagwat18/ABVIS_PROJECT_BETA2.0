"""Production / engineering screens of the one desktop app (gui.App): History, Health, Models, and
the two engineering dialogs of the inspection line (speed calibration, camera stations / line layout).

Same rules as gui.py: every tab takes (app, parent), keeps self.app and has refresh(); widgets are
touched on the Tk thread only; slow work goes through app.run_bg. Nothing here talks to a device:
the PLC is read through app.plc (cached state), cameras through app.cams, history through
app.store() (production_store), alarms through app.alarms.
"""
from __future__ import annotations

import csv
import os
import shutil
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from line import alarms as AL
from ui import charts
from vision import dataset as D
from line import machine_cycle as MC
from registry import model_registry as MR
from ui import theme
from line import tracking as TR
from vision import verdict as V
from ui.theme import (ACC, ACC_H, ACC_SOFT, ACC_T, BAD, DIM, FIELD, GOOD, INK, LINE, MONO, OFF, PANEL, PANEL_2,
                   WARN)

SEV_COLOUR = {AL.CRITICAL: BAD, AL.MAJOR: BAD, AL.WARNING: WARN, AL.INFO: DIM}


def box(parent, title, **pack):
    f = ctk.CTkFrame(parent, fg_color=PANEL, border_width=1, border_color=LINE)
    f.pack(**(pack or {"fill": "x", "pady": (0, 8)}))
    ctk.CTkLabel(f, text=title, font=theme.CAPS, text_color=DIM).pack(anchor="w", padx=12, pady=(8, 2))
    return f


def listbox(parent, height=12):
    """A selectable row list in the theme (CTk has none). Monospace so columns line up."""
    lb = tk.Listbox(parent, font=MONO, bg=FIELD, fg=INK, selectbackground=ACC_SOFT, selectforeground=INK,
                    highlightthickness=1, highlightbackground=LINE, highlightcolor=ACC, bd=0, activestyle="none",
                    height=height, exportselection=False)
    return lb


def _f(v, fmt="{:.3f}", none="--"):
    return none if v is None else fmt.format(v)


# =============================================================================== dialogs
class SpeedCalibrationDialog(ctk.CTkToplevel):
    """Time-based conveyor speed calibration (this machine has no encoder).

    Measure a distance on the belt (two marks), run the conveyor, and time one bottle from mark A to
    mark B several times -- with the stopwatch buttons or by typing measured times. The mean speed is
    saved with its spread and a timestamp; a spread above the tolerance is refused, because time-based
    tracking would then mis-time the reject."""

    def __init__(self, app, on_saved=None):
        super().__init__(app)
        self.app, self.on_saved = app, on_saved
        self.title("Conveyor speed calibration (time-based, no encoder)")
        self.geometry("620x640")
        self.transient(app)
        s = app.settings
        self.times: list = []
        self._t0 = None
        self.rec = None
        f = box(self, "1  MEASURE A DISTANCE ON THE BELT", fill="x", padx=12, pady=(12, 6))
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(row, text="Mark A -> mark B (mm)").pack(side="left")
        self.dist = ctk.CTkEntry(row, width=90)
        self.dist.pack(side="left", padx=6)
        ctk.CTkLabel(row, text="Tolerance %").pack(side="left", padx=(16, 4))
        self.tol = ctk.CTkEntry(row, width=60)
        self.tol.insert(0, f"{float(s.get('speed_tolerance_pct', TR.TRACKING_DEFAULTS['speed_tolerance_pct'])):g}")
        self.tol.pack(side="left")
        f = box(self, "2  RUN THE CONVEYOR AND TIME ONE BOTTLE, AT LEAST 3 TIMES", fill="x", padx=12, pady=6)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 6))
        self.b_go = ctk.CTkButton(row, text="Bottle at mark A  (start)", width=200, fg_color=ACC, text_color=ACC_T,
                                  hover_color=ACC_H, command=self.start_watch)
        self.b_go.pack(side="left")
        self.b_stop = ctk.CTkButton(row, text="Bottle at mark B  (stop)", width=200, state="disabled",
                                    command=self.stop_watch)
        self.b_stop.pack(side="left", padx=8)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(row, text="or type a measured time (s)").pack(side="left")
        self.manual = ctk.CTkEntry(row, width=80)
        self.manual.pack(side="left", padx=6)
        ctk.CTkButton(row, text="Add", width=60, command=self.add_manual).pack(side="left")
        ctk.CTkButton(row, text="Remove last", width=100, command=self.remove_last).pack(side="left", padx=8)
        self.runs = ctk.CTkLabel(f, text="runs: -", font=MONO, anchor="w", justify="left")
        self.runs.pack(fill="x", padx=12, pady=(0, 10))
        f = box(self, "3  RESULT", fill="both", expand=True, padx=12, pady=6)
        self.out = ctk.CTkLabel(f, text="", font=MONO, anchor="nw", justify="left")
        self.out.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text="Calculate", width=110, command=self.calculate).pack(side="left")
        self.b_save = ctk.CTkButton(row, text="Save speed", width=120, fg_color=ACC, text_color=ACC_T,
                                    hover_color=ACC_H, state="disabled", command=self.save)
        self.b_save.pack(side="left", padx=8)
        ctk.CTkButton(row, text="Close", width=80, command=self.destroy).pack(side="right")
        self.after(100, self.lift)

    def start_watch(self):
        self._t0 = time.monotonic()
        self.b_go.configure(state="disabled")
        self.b_stop.configure(state="normal", fg_color=GOOD, text_color=ACC_T)

    def stop_watch(self):
        if self._t0 is not None:
            self.times.append(round(time.monotonic() - self._t0, 3))
        self._t0 = None
        self.b_go.configure(state="normal")
        self.b_stop.configure(state="disabled", fg_color=PANEL_2, text_color=INK)
        self._show_runs()

    def add_manual(self):
        try:
            v = float(self.manual.get())
            if v <= 0:
                raise ValueError
        except ValueError:
            return messagebox.showerror("Time", "Type the measured time in seconds (> 0).", parent=self)
        self.times.append(v)
        self.manual.delete(0, "end")
        self._show_runs()

    def remove_last(self):
        if self.times:
            self.times.pop()
        self._show_runs()

    def _show_runs(self):
        self.runs.configure(text="runs: " + ("  ".join(f"{t:.3f}s" for t in self.times) or "-"))
        self.b_save.configure(state="disabled")

    def calculate(self):
        try:
            dist, tol = float(self.dist.get()), float(self.tol.get())
        except ValueError:
            return messagebox.showerror("Calibration", "Distance and tolerance must be numbers.", parent=self)
        self.rec = rec = TR.calibrate_speed(dist, self.times, tol)
        lines = []
        if "mean_mm_s" in rec:
            lines += [f"mean speed     {rec['mean_mm_s']:.2f} mm/s",
                      f"min / max      {rec['min_mm_s']:.2f} / {rec['max_mm_s']:.2f} mm/s",
                      f"spread         {rec['spread_pct']:.2f} %   (limit {tol:g} %)",
                      f"mean time      {rec['mean_travel_s']:.3f} s over {dist:g} mm"]
            cfg = MC.line_settings(dict(self.app.settings, conveyor_mm_s=rec["mean_mm_s"], speed_tolerance_pct=tol))
            d = float(cfg.get("inspection_to_reject_mm") or 0)
            if d > 0:
                src = TR.position_source(cfg)
                travel, unc = src.eta_s(d), src.uncertainty_s(d)
                t0 = float(cfg["plc_t0_s"])
                lines += ["", f"inspection->reject {d:g} mm:",
                          f"  travel         {travel:.3f} s  +- {unc:.3f} s",
                          f"  REJECT sent at trigger + {travel - t0:.3f} s  (travel - T0 {t0:g} s)",
                          f"  margin left    {travel - t0 - float(cfg['inspect_window_s']):.3f} s after the frame window"]
            else:
                lines += ["", "inspection->reject distance not set yet (Line layout)."]
        if rec.get("problem"):
            lines += ["", f"NOT SAVABLE: {rec['problem']}"]
        self.out.configure(text="\n".join(lines), text_color=BAD if rec.get("problem") else INK)
        self.b_save.configure(state="disabled" if rec.get("problem") else "normal")

    def save(self):
        if not self.rec or self.rec.get("problem"):
            return
        s = self.app.settings
        s["conveyor_mm_s"] = self.rec["mean_mm_s"]
        s["speed_tolerance_pct"] = self.rec["tolerance_pct"]
        s["speed_calibration"] = self.rec
        D.save_settings(s)
        if self.on_saved:
            self.on_saved()
        messagebox.showinfo("Saved", f"Conveyor speed {self.rec['mean_mm_s']:.2f} mm/s saved "
                                     f"({self.rec['time']}).", parent=self)


class LineLayoutDialog(ctk.CTkToplevel):
    """Where things are on the belt: inspection->reject distance and each line camera's station
    (offset downstream of the trigger photo-eye, wall side, role, per-camera decision overrides).
    Saved to settings.json; checked with tracking.station_problems before it is saved."""

    def __init__(self, app, cameras, on_saved=None):
        super().__init__(app)
        self.app, self.cameras, self.on_saved = app, list(cameras), on_saved   # [(source, label)]
        self.title("Line layout - camera stations and distances")
        self.geometry("980x520")
        self.transient(app)
        s = MC.line_settings(app.settings)
        f = box(self, "BELT", fill="x", padx=12, pady=(12, 6))
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))
        self.ent = {}
        for key, label, w in (("inspection_to_reject_mm", "Photo-eye (camera 1) -> reject cylinder, mm", 80),
                              ("timing_margin_s", "Timing margin s", 60)):
            ctk.CTkLabel(row, text=label).pack(side="left", padx=(0, 4))
            e = ctk.CTkEntry(row, width=w)
            e.insert(0, f"{float(s.get(key) or 0):g}")
            e.pack(side="left", padx=(0, 16))
            self.ent[key] = e
        ctk.CTkLabel(row, text=f"speed {float(s.get('conveyor_mm_s') or 0):g} mm/s "
                               f"({'calibrated' if s.get('speed_calibration') else 'not calibrated'})",
                     text_color=DIM).pack(side="left")
        f = box(self, "CAMERA STATIONS (offset = distance DOWNSTREAM of the trigger photo-eye; 0 = at it)",
                fill="x", padx=12, pady=6)
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 10))
        heads = ("Camera", "Name", "Role", "Offset mm", "Wall / side", "station_x (0-1)", "Judges parts (blank = all)")
        for c, h in enumerate(heads):
            ctk.CTkLabel(grid, text=h, font=theme.CAPS, text_color=DIM).grid(row=0, column=c, sticky="w", padx=4)
        self.rows = {}
        st = s.get("camera_stations") or {}
        for r, (src, label) in enumerate(self.cameras, 1):
            cur = st.get(str(src)) or {}
            rules = cur.get("rules") or {}
            ctk.CTkLabel(grid, text=label[:28]).grid(row=r, column=0, sticky="w", padx=4, pady=2)
            w = {"name": ctk.CTkEntry(grid, width=110), "role": ctk.CTkOptionMenu(grid, values=list(TR.ROLES), width=140),
                 "offset_mm": ctk.CTkEntry(grid, width=80), "side": ctk.CTkEntry(grid, width=100),
                 "station_x": ctk.CTkEntry(grid, width=70), "judge": ctk.CTkEntry(grid, width=170)}
            w["name"].insert(0, cur.get("name") or f"Camera {r}")
            w["role"].set(cur.get("role") or "general")
            w["offset_mm"].insert(0, f"{float(cur.get('offset_mm') or 0):g}")
            w["side"].insert(0, cur.get("side") or "")
            if "station_x" in rules:
                w["station_x"].insert(0, f"{rules['station_x']:g}")
            if rules.get("judge"):
                w["judge"].insert(0, ", ".join(rules["judge"]))
            for c, k in enumerate(("name", "role", "offset_mm", "side", "station_x", "judge"), 1):
                w[k].grid(row=r, column=c, sticky="w", padx=4, pady=2)
            self.rows[str(src)] = w
        if not self.cameras:
            ctk.CTkLabel(grid, text="No line camera chosen on the Production screen.", text_color=WARN).grid(
                row=1, column=0, columnspan=7, sticky="w")
        self.msg = ctk.CTkLabel(self, text="", anchor="w", justify="left", wraplength=940)
        self.msg.pack(fill="x", padx=14, pady=4)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text="Check", width=90, command=self.check).pack(side="left")
        ctk.CTkButton(row, text="Save", width=90, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.save).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Close", width=80, command=self.destroy).pack(side="right")
        self.after(100, self.lift)

    def values(self) -> dict:
        out = {}
        for key, e in self.ent.items():
            try:
                v = float(e.get() or 0)
            except ValueError:
                raise ValueError(f"{key}: not a number") from None
            if v < 0:
                raise ValueError(f"{key}: must be >= 0")
            out[key] = v
        stations = {}
        for src, w in self.rows.items():
            try:
                off = float(w["offset_mm"].get() or 0)
            except ValueError:
                raise ValueError(f"camera {src}: offset must be a number (mm)") from None
            rules = {}
            sx = w["station_x"].get().strip()
            if sx:
                try:
                    rules["station_x"] = float(sx)
                except ValueError:
                    raise ValueError(f"camera {src}: station_x must be a number 0-1") from None
                if not 0 <= rules["station_x"] <= 1:
                    raise ValueError(f"camera {src}: station_x must be 0-1")
            judge = [p.strip() for p in w["judge"].get().split(",") if p.strip()]
            if judge:
                rules["judge"] = judge
            stations[src] = {"name": w["name"].get().strip(), "role": w["role"].get(), "offset_mm": off,
                             "side": w["side"].get().strip(), "rules": rules}
        out["camera_stations"] = stations
        return out

    def check(self) -> bool:
        try:
            vals = self.values()
        except ValueError as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return False
        cfg = MC.line_settings(dict(self.app.settings, **vals))
        probs = MC.line_problems(cfg, [src for src, _ in self.cameras])
        if probs:
            self.msg.configure(text="Line will not start:\n- " + "\n- ".join(probs), text_color=BAD)
            return False
        offs = TR.station_offsets_s(cfg, [src for src, _ in self.cameras])
        self.msg.configure(text="OK.  Bottle reaches: " + ",  ".join(
            f"{vals['camera_stations'][k]['name'] or k} +{v:.2f} s" for k, v in offs.items())
            + f"   |   {TR.position_source(cfg).describe()}", text_color=GOOD)
        return True

    def save(self):
        try:
            vals = self.values()
        except ValueError as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return
        ok = self.check()
        self.app.settings.update(vals)
        D.save_settings(self.app.settings)
        if self.on_saved:
            self.on_saved()
        if ok:
            self.msg.configure(text=self.msg.cget("text") + "   - saved.")
        else:
            self.msg.configure(text=self.msg.cget("text") + "\n(saved, but the line will not start until this is fixed)")


# =============================================================================== History
class HistoryTab:
    """Persistent production history (production_store): every bottle of a day, its result, defects,
    PLC outcome and evidence, the day's counts / defect distribution / hourly throughput, and alarms."""

    FILTERS = ("All", "GOOD", "DEFECT", "FAULT")

    def __init__(self, app, parent):
        self.app = app
        self.rows: list = []
        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(bar, text="Day").pack(side="left", padx=(12, 4), pady=8)
        self.day = ctk.CTkOptionMenu(bar, values=["-"], width=130, command=lambda _=None: self.refresh())
        self.day.pack(side="left")
        ctk.CTkLabel(bar, text="Result").pack(side="left", padx=(14, 4))
        self.filter = ctk.CTkOptionMenu(bar, values=list(self.FILTERS), width=100, command=lambda _=None: self.refresh())
        self.filter.pack(side="left")
        ctk.CTkLabel(bar, text="Shift").pack(side="left", padx=(14, 4))
        self.shift = ctk.CTkOptionMenu(bar, values=["Whole day"] + [s[0] for s in self.shifts()], width=110,
                                       command=lambda _=None: self.refresh())
        self.shift.pack(side="left")
        ctk.CTkButton(bar, text="Refresh", width=80, command=self.refresh).pack(side="left", padx=8)
        ctk.CTkButton(bar, text="Open evidence", width=120, command=self.open_evidence).pack(side="left")
        ctk.CTkButton(bar, text="Export day CSV", width=120, command=self.export).pack(side="left", padx=8)
        self.where = ctk.CTkLabel(bar, text="", text_color=DIM, font=theme.TINY)
        self.where.pack(side="right", padx=12)
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=(0, 8))
        cnt = ctk.CTkFrame(top, fg_color=PANEL, border_width=1, border_color=LINE)
        cnt.pack(side="left", fill="y", padx=(0, 8))
        self.cnt = {}
        for k, title, col in (("total", "TOTAL", INK), ("PASS", "GOOD", GOOD), ("REJECT", "DEFECT", BAD),
                              ("FAULT", "FAULT", WARN), ("yield", "GOOD %", INK)):
            cell = ctk.CTkFrame(cnt, fg_color="transparent")
            cell.pack(side="left", padx=12, pady=10)
            self.cnt[k] = ctk.CTkLabel(cell, text="0", font=theme.BIG, text_color=col)
            self.cnt[k].pack()
            ctk.CTkLabel(cell, text=title, text_color=DIM, font=theme.CAPS).pack()
        self.c_def = self._chart(top, "DEFECTS (this day)")
        self.c_hour = self._chart(top, "BOTTLES PER HOUR")
        low = ctk.CTkFrame(parent, fg_color="transparent")
        low.pack(fill="both", expand=True)
        left = box(low, "INSPECTIONS (newest first)", side="left", fill="both", expand=True, padx=(0, 8))
        head = f"{'TIME':<9}{'ID':<8}{'RESULT':<8}{'DEFECTS':<26}{'CONF':>5}  {'CMD':<7}{'PLC':<26}EVIDENCE"
        ctk.CTkLabel(left, text=head, font=MONO, text_color=DIM, anchor="w").pack(fill="x", padx=12)
        self.list = listbox(left, 16)
        self.list.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.list.bind("<Double-Button-1>", lambda e: self.open_evidence())
        right = box(low, "ALARMS (this day)", side="left", fill="both", padx=0)
        self.alarm_list = listbox(right, 16)
        self.alarm_list.configure(width=58)
        self.alarm_list.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def shifts(self) -> list:
        """settings.json "shifts": [[name, "HH:MM", "HH:MM"], ...]; default A 06-14, B 14-22, C 22-06."""
        from line import production_store
        raw = self.app.settings.get("shifts") or production_store.DEFAULT_SHIFTS
        return [tuple(x) for x in raw if len(x) == 3]

    def span(self, day):
        from line import production_store
        sh = next((x for x in self.shifts() if x[0] == self.shift.get()), None)
        return None if sh is None else production_store.shift_span(day, sh)

    def _chart(self, parent, title):
        f = box(parent, title, side="left", fill="both", expand=True, padx=(0, 8))
        cv = ctk.CTkCanvas(f, width=360, height=150, bg=PANEL, highlightthickness=0)
        cv.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        return cv

    def refresh(self):
        st = self.app.store()
        days = st.days() or [time.strftime("%Y-%m-%d")]
        cur = self.day.get()
        self.day.configure(values=days)
        if cur not in days:
            self.day.set(days[0])
        day = self.day.get()
        f = {"GOOD": "PASS", "DEFECT": "REJECT"}.get(self.filter.get(), self.filter.get())
        span = self.span(day)
        self.rows = st.recent(1000, day=None if span else day, final=None if f == "All" else f, span=span)
        s = st.summary_range(*span, label=f"{day} shift {self.shift.get()}") if span else st.summary(day)
        for k in ("total", "PASS", "REJECT", "FAULT"):
            self.cnt[k].configure(text=str(s[k]))
        self.cnt["yield"].configure(text=f"{100 * s['PASS'] / s['total']:.1f}" if s["total"] else "--")
        defs = list(s["defects"].items())[:8]
        charts.bar_chart(self.c_def, [d for d, _ in defs], [v for _, v in defs], colours=[BAD] * len(defs))
        hours = sorted(s["hourly"])
        charts.bar_chart(self.c_hour, [f"{h:02d}" for h in hours], [sum(s["hourly"][h].values()) for h in hours])
        self.list.delete(0, "end")
        for r in self.rows:
            d = r["data"]
            conf = "" if r["confidence"] is None else f"{r['confidence']:.2f}"
            line = (f"{time.strftime('%H:%M:%S', time.localtime(r['wall'])):<9}{r['inspection_id']:<8}{V.shown_result(r['final']):<8}"
                    f"{(', '.join(V.pretty(x) for x in (r['defects'] or '').split(';') if x) or ('-' if r['final'] == 'PASS' else d.get('reason', '')))[:25]:<26}{conf:>5}  "
                    f"{(r['command'] or '-'):<7}{(r['plc_status'] or '')[:25]:<26}{'yes' if r['evidence'] else ''}")
            self.list.insert("end", line)
            self.list.itemconfig("end", fg={"PASS": GOOD, "REJECT": BAD}.get(r["final"], WARN))
        self.alarm_list.delete(0, "end")
        for a in st.alarms(day):
            self.alarm_list.insert("end", f"{time.strftime('%H:%M:%S', time.localtime(a['wall']))} {a['state'][:5]:<6}"
                                          f"{a['code']:<26}{a['cause'][:60]}")
            self.alarm_list.itemconfig("end", fg=SEV_COLOUR.get(a["severity"], DIM) if a["state"] == "ACTIVE" else DIM)
        self.where.configure(text=f"{st.path}")

    def open_evidence(self):
        sel = self.list.curselection()
        if not sel:
            return messagebox.showinfo("Evidence", "Select an inspection first.")
        r = self.rows[sel[0]]
        if not r["evidence"]:
            return messagebox.showinfo("Evidence", f"No evidence image kept for {r['inspection_id']} "
                                                   f"(policy: {self.app.settings.get('evidence_policy', 'REJECT_AND_FAULT')}).")
        p = self.app.store().folder / r["evidence"]
        if not p.exists():
            return messagebox.showwarning("Evidence", f"{p} is missing.")
        try:
            os.startfile(str(p))                                         # noqa: S606 - local file, Windows viewer
        except (AttributeError, OSError) as e:
            messagebox.showinfo("Evidence", f"{p}\n({e})")

    def export(self):
        if not self.rows:
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"production_{self.day.get()}.csv")
        if not out:
            return
        keys = list(self.rows[0]["data"])
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "evidence"] + keys)
            w.writeheader()
            for r in reversed(self.rows):
                w.writerow({"run_id": r["run_id"], "evidence": r["evidence"], **r["data"]})


# =============================================================================== Health
class HealthTab:
    """One page that says whether the station is healthy: computer, PLC, cameras, models, storage,
    alarms. HEALTHY / WARNING / FAULT per item. Cached state only (no device I/O)."""

    def __init__(self, app, parent):
        self.app = app
        self.cards = {}
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        names = ("COMPUTER", "PLC", "PLC I/O", "CAMERAS", "MODELS", "STORAGE / LOGS", "ALARMS")
        for i, n in enumerate(names):
            f = ctk.CTkFrame(grid, fg_color=PANEL, border_width=1, border_color=LINE)
            f.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0, 8), pady=(0, 8))
            grid.grid_columnconfigure(i % 3, weight=1, uniform="h")
            grid.grid_rowconfigure(i // 3, weight=1)
            head = ctk.CTkFrame(f, fg_color="transparent")
            head.pack(fill="x", padx=12, pady=(10, 4))
            ctk.CTkLabel(head, text=n, font=theme.CAPS, text_color=DIM).pack(side="left")
            badge = ctk.CTkLabel(head, text="--", font=theme.CAPS, text_color=ACC_T, fg_color=OFF, corner_radius=6,
                                 width=90)
            badge.pack(side="right")
            body = ctk.CTkLabel(f, text="", font=MONO, anchor="nw", justify="left")
            body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
            self.cards[n] = (badge, body)
        from line import applog
        lg = box(parent, "LOGS  (logs/*.log: one line per event, newest first)", fill="both", expand=True)
        row = ctk.CTkFrame(lg, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 4))
        self.log_ch = ctk.CTkOptionMenu(row, values=["all"] + list(applog.CHANNELS), width=120,
                                        command=lambda _=None: self.show_logs())
        self.log_ch.pack(side="left")
        self.log_lv = ctk.CTkOptionMenu(row, values=["any level", "INFO", "WARNING", "ERROR", "CRITICAL"], width=120,
                                        command=lambda _=None: self.show_logs())
        self.log_lv.pack(side="left", padx=6)
        self.log_q = ctk.CTkEntry(row, width=260, placeholder_text="search text (e.g. CAMERA, 000123, COM5)")
        self.log_q.pack(side="left", padx=6)
        self.log_q.bind("<Return>", lambda e: self.show_logs())
        ctk.CTkButton(row, text="Search", width=80, command=self.show_logs).pack(side="left")
        self.log_where = ctk.CTkLabel(row, text=str(applog.folder()), text_color=DIM, font=theme.TINY)
        self.log_where.pack(side="right")
        self.log_box = ctk.CTkTextbox(lg, font=("Consolas", 12), wrap="none", height=160)
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self._loop_on = False

    def show_logs(self):
        from line import applog
        ch = self.log_ch.get()
        lv = self.log_lv.get()
        lines = applog.search(None if ch == "all" else ch, self.log_q.get().strip(),
                              None if lv == "any level" else lv, limit=400)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", "\n".join(lines) or "no matching log line")
        self.log_box.configure(state="disabled")

    def refresh(self):
        self.update()
        self.show_logs()
        if not self._loop_on:
            self._loop_on = True
            self.app.after(2000, self._loop)

    def _loop(self):
        if self.app.tabs.get() == "Health":
            try:
                self.update()
            except Exception:                                       # noqa: BLE001 - display only
                import traceback
                traceback.print_exc()
            self.app.after(2000, self._loop)
        else:
            self._loop_on = False

    def _set(self, name, level, text):
        badge, body = self.cards[name]
        badge.configure(text=level, fg_color={"HEALTHY": GOOD, "WARNING": WARN, "FAULT": BAD}.get(level, OFF))
        body.configure(text=text)

    def update(self):
        from vision import bench
        s = bench.system_stats()
        lvl = "HEALTHY"
        if (s.get("ram_pct") or 0) > 90 or (s.get("cpu_sys") or 0) > 95:
            lvl = "WARNING"
        gpu = s.get("gpu_name") or "no CUDA GPU (CPU inference)"
        vram = (f"{s['gpu_mem_mb']} / {s['gpu_mem_total_mb']} MB" if s.get("gpu_mem_total_mb") else "--")
        self._set("COMPUTER", lvl, f"CPU system   {_f(s.get('cpu_sys'), '{:.0f} %')}\nCPU app      "
                                    f"{_f(s.get('cpu_proc'), '{:.0f} %')}\nRAM          {_f(s.get('ram_pct'), '{:.0f} %')}"
                                    f"  (app {_f(s.get('ram_mb'), '{} MB')})\nGPU          {gpu}\nVRAM         {vram}\n"
                                    f"GPU util     {_f(s.get('gpu_util'), '{} %')}")
        svc = self.app.plc
        h = svc.health_check()
        snap = svc.snapshot() if h["link_state"] == "CONNECTED" else None
        run = None if not snap else bool(snap["plc_run"])
        lvl = "HEALTHY" if h["link_state"] == "CONNECTED" and run else ("FAULT" if h["link_state"] == "FAULT"
                                                                       else "WARNING")
        lat = h.get("last_latency_ms")
        self._set("PLC", lvl, f"link         {h['link_state']}\nmode         "
                              f"{'SIMULATOR' if svc.simulator_mode else 'REAL PLC'}\nendpoint     "
                              f"{svc.client.transport.description}\nPLC          "
                              f"{'RUN' if run else ('STOP' if run is False else '--')}\nlatency      "
                              f"{_f(lat, '{:.0f} ms')}\nreplies      {h.get('ok')}\nlast error   "
                              f"{(h.get('last_error') or '-')[:60]}")
        # every PLC input / output / command bit, read-only, from the service's cached status read
        labels = (("X0", "photo-eye"), ("X1", "START button"), ("X2", "STOP button"), ("Y1", "conveyor motor"),
                  ("Y0", "reject valve"), ("M2", "trigger (inspect now)"), ("M0", "PASS command"),
                  ("M1", "REJECT command"), ("M10", "START test bit"), ("M11", "STOP test bit"))
        vals = {}
        if snap and snap.get("age_s", 9) <= 1.5:
            for grp in ("inputs", "outputs", "internal"):
                vals.update(snap.get(grp) or {})
            if "m2" in snap:
                vals.setdefault("M2", snap["m2"])
        rows = [f"{k:<4}{'--' if k not in vals else ('ON ' if vals[k] else 'off'):<5}{txt}" for k, txt in labels]
        self._set("PLC I/O", "HEALTHY" if vals else "WARNING",
                  ("live from the PLC (read only)\n" if vals else "no live PLC data: connect first\n") + "\n".join(rows))
        cams = self.app.cams.running()
        lines, lvl = [], "HEALTHY" if cams else "WARNING"
        for c in cams:
            age = None if c.frame_ts is None else time.monotonic() - c.frame_ts
            bad = bool(c.error) or not c.alive or age is None or age > 1.0
            lvl = "FAULT" if bad else lvl
            wh = "x".join(map(str, c.frame_wh)) if c.frame_wh else "--"
            lines.append(f"{c.name[:22]:<23}{wh:<10}{c.fps:5.1f} fps  age {_f(age, '{:.2f}s')}"
                         + (f"\n   ERROR {c.error[:50]}" if c.error else ""))
        self._set("CAMERAS", lvl, "\n".join(lines) or "no camera streaming\n(cameras open only when a test or the line "
                                                       "starts)")
        cfg = D.load_config()
        stamp = cfg.get("active_model")
        cls_ok = bool(stamp) and (D.MODELS / str(stamp) / "model.pt").exists()
        from vision import detect
        w = Path(self.app.settings.get("detector_weights") or detect.DEFAULT_WEIGHTS)
        lvl = "HEALTHY" if cls_ok and w.exists() else "WARNING"
        try:
            from line import production_store
            rid = production_store.recipe_id(cfg.get("inspection"))
        except Exception:                                           # noqa: BLE001
            rid = "?"
        self._set("MODELS", lvl, f"project      {D.PROJECT}\nclassifier   {stamp or 'NONE'} "
                                 f"{'' if cls_ok else '(missing)'}\ndetector     {w.name} {'' if w.exists() else '(missing)'}"
                                 f"\nrecipe       {rid}\nthresholds   {len(cfg.get('thresholds', {}))} defects\n"
                                 f"job          {self.app.settings.get('job_id') or '-'}")
        st = self.app.store()
        try:
            free = shutil.disk_usage(str(st.folder)).free / 2 ** 30
        except OSError:
            free = None
        size = st.path.stat().st_size / 2 ** 20 if st.path.exists() else 0
        lvl = "FAULT" if free is not None and free < 1 else ("WARNING" if free is not None and free < 5 else "HEALTHY")
        self.app.alarms.set_condition("DISK_LOW", free is not None and free < 1, f"{free:.1f} GB free" if free else "")
        self._set("STORAGE / LOGS", lvl, f"disk free    {_f(free, '{:.1f} GB')}\ndatabase     {size:.1f} MB\n"
                                         f"folder       {st.folder}\nevidence     written {st.written}, dropped "
                                         f"{st.dropped}\npolicy       {self.app.settings.get('evidence_policy', 'REJECT_AND_FAULT')}")
        act = self.app.alarms.active()
        worst = self.app.alarms.worst()
        lvl = "FAULT" if worst in (AL.CRITICAL, AL.MAJOR) else ("WARNING" if act else "HEALTHY")
        self._set("ALARMS", lvl, "\n".join(f"{a.severity[:4]} {a.code} x{a.count}" for a in act[:8]) or "no active alarm")


# =============================================================================== Models
class ModelsTab:
    """Model registry (model_registry.py): every trained classifier / detector found on disk, its status
    (CANDIDATE / VALIDATED / APPROVED / ACTIVE / ARCHIVED / REJECTED), held-out test results, the
    critical classes, and the actions validate -> approve -> activate, reject, rollback."""

    def __init__(self, app, parent):
        self.app = app
        self.reg = MR.Registry()
        self.entries: list = []
        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        for text, fn, primary in (("Refresh", self.refresh, False), ("Validate...", self.validate, False),
                                  ("Approve", self.approve, False), ("Reject", self.reject, False),
                                  ("ACTIVATE", self.activate, True), ("Roll back classifier", lambda: self.rollback("classification"), False),
                                  ("Roll back detector", lambda: self.rollback("detection"), False)):
            kw = dict(fg_color=ACC, text_color=ACC_T, hover_color=ACC_H) if primary else {}
            ctk.CTkButton(bar, text=text, width=max(90, 9 * len(text)), command=fn, **kw).pack(side="left", padx=(8, 0),
                                                                                              pady=8)
        ctk.CTkLabel(bar, text="Training never activates a model: validate on the real camera, approve, then activate.",
                     text_color=DIM, font=theme.TINY).pack(side="right", padx=12)
        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        left = box(body, "MODELS", side="left", fill="both", expand=True, padx=(0, 8))
        ctk.CTkLabel(left, text=f"{'STATUS':<11}{'MODEL':<40}{'ARCH':<17}{'TEST':<6}WARNINGS", font=MONO,
                     text_color=DIM, anchor="w").pack(fill="x", padx=12)
        self.list = listbox(left, 18)
        self.list.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.list.bind("<<ListboxSelect>>", lambda e: self.show())
        right = box(body, "DETAILS", side="left", fill="both", expand=True)
        self.detail = ctk.CTkTextbox(right, font=MONO, wrap="word")
        self.detail.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def refresh(self):
        try:
            self.entries = self.reg.discover()
        except Exception as e:                                         # noqa: BLE001 - shown
            self.entries = []
            self.detail.delete("1.0", "end")
            self.detail.insert("end", f"registry error: {type(e).__name__}: {e}")
        self.list.delete(0, "end")
        for e in self.entries:
            self.list.insert("end", f"{e['status']:<11}{e['name'][:39]:<40}{str(e.get('arch'))[:16]:<17}"
                                    f"{'yes' if e['has_test'] else 'NO':<6}{'; '.join(e['warnings'])[:80]}")
            self.list.itemconfig("end", fg={MR.ACTIVE: GOOD, MR.REJECTED: DIM, MR.ARCHIVED: DIM}.get(
                e["status"], WARN if e["warnings"] else INK))

    def selected(self):
        sel = self.list.curselection()
        if not sel:
            messagebox.showinfo("Models", "Select a model first.")
            return None
        return self.entries[sel[0]]

    def show(self):
        sel = self.list.curselection()
        if not sel:
            return
        e = self.entries[sel[0]]
        import json
        txt = {k: v for k, v in e.items() if k not in ("history", "thresholds")}
        self.detail.delete("1.0", "end")
        self.detail.insert("end", json.dumps(txt, indent=2, default=str))
        if e.get("history"):
            self.detail.insert("end", "\n\nHISTORY\n" + "\n".join(
                f"{h['time']}  {h['status']:<10} {h['by']}: {h['note']}" for h in e["history"]))

    def _do(self, fn, ok_msg):
        try:
            fn()
        except MR.RegistryError as e:
            return messagebox.showwarning("Not allowed", str(e))
        self.app.settings = D.load_settings()          # activation may have changed detector_weights on disk
        self.refresh()
        if ok_msg:
            messagebox.showinfo("Models", ok_msg)

    def validate(self):
        e = self.selected()
        if e:
            ValidateDialog(self.app, self.reg, e, on_done=self.refresh)

    def approve(self):
        e = self.selected()
        if e:
            self._do(lambda: self.reg.approve(e["name"]), "")

    def reject(self):
        e = self.selected()
        if e and messagebox.askyesno("Reject", f"Mark {e['name']} REJECTED?"):
            self._do(lambda: self.reg.reject(e["name"]), "")

    def _line_busy(self) -> bool:
        if self.app.tab_production.running:
            messagebox.showwarning("Line running", "Stop the inspection line before changing the production model.")
            return True
        return False

    def activate(self):
        e = self.selected()
        if not e or self._line_busy():
            return
        if not messagebox.askyesno("Activate", f"Make {e['name']} the production {e['kind']} model?\n\n"
                                               f"The current one is archived and can be rolled back."):
            return
        self._do(lambda: self.reg.activate(e["name"]), f"{e['name']} is now ACTIVE.")
        self.app.reload()

    def rollback(self, kind):
        if self._line_busy():
            return
        if messagebox.askyesno("Roll back", f"Re-activate the previous {kind} model?"):
            self._do(lambda: self.reg.rollback(kind), "Rolled back.")
            self.app.reload()


# =============================================================================== recipe editor
class RecipeDialog(ctk.CTkToplevel):
    """The project's inspection recipe (config.json "inspection") as a form: the anchor class (the product)
    and up to six parts that must belong to it, where to look for each and where it must sit.

    Read by decision.detection_findings on every bottle. Checked with decision.recipe_problems and against
    the detector's classes before it is saved; refused while the line runs (every run records the recipe
    hash it used, production_store.recipe_id)."""

    ROWS = 6

    def __init__(self, app, classes=(), on_saved=None):
        super().__init__(app)
        from line import decision as DEC
        self.DEC, self.app, self.on_saved = DEC, app, on_saved
        self.classes = tuple(classes)
        self.title(f"Inspection recipe - {D.project_title(D.PROJECT)}")
        self.geometry("1000x520")
        self.transient(app)
        cur = D.load_config().get("inspection") or DEC.default_recipe(DEC.RULES)
        f = box(self, "PRODUCT", fill="x", padx=12, pady=(12, 6))
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(row, text="Anchor class (the inspected object; none found = FAULT)").pack(side="left")
        self.anchor = ctk.CTkEntry(row, width=140)
        self.anchor.insert(0, cur.get("anchor", ""))
        self.anchor.pack(side="left", padx=8)
        ctk.CTkLabel(row, text=f"detector classes: {', '.join(self.classes) or 'unknown'}", text_color=DIM).pack(
            side="left", padx=12)
        f = box(self, "PARTS  (positions are fractions of the anchor height from its top; -0.25 = a quarter above it)",
                fill="x", padx=12, pady=6)
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 10))
        for c, h in enumerate(("Part class", "Required", "Search top", "Search bottom", "Zone top", "Zone bottom",
                               "Defect if outside zone")):
            ctk.CTkLabel(grid, text=h, font=theme.CAPS, text_color=DIM).grid(row=0, column=c, sticky="w", padx=4)
        self.rows = []
        parts = list(cur.get("parts") or [])
        for r in range(self.ROWS):
            p = parts[r] if r < len(parts) else {}
            w = {"name": ctk.CTkEntry(grid, width=110), "required": ctk.CTkCheckBox(grid, text="", width=30),
                 "s0": ctk.CTkEntry(grid, width=70), "s1": ctk.CTkEntry(grid, width=70),
                 "z0": ctk.CTkEntry(grid, width=70), "z1": ctk.CTkEntry(grid, width=70),
                 "misplaced": ctk.CTkEntry(grid, width=160)}
            if p:
                w["name"].insert(0, p.get("name", ""))
                if p.get("required", True):
                    w["required"].select()
                for k, (a, b) in (("s", p.get("search") or (None, None)), ("z", p.get("zone") or (None, None))):
                    if a is not None:
                        w[k + "0"].insert(0, f"{a:g}")
                        w[k + "1"].insert(0, f"{b:g}")
                if p.get("misplaced"):
                    w["misplaced"].insert(0, p["misplaced"])
            for c, k in enumerate(("name", "required", "s0", "s1", "z0", "z1", "misplaced")):
                w[k].grid(row=r + 1, column=c, sticky="w", padx=4, pady=2)
            self.rows.append(w)
        self.msg = ctk.CTkLabel(self, text="A missing required part -> defect missing_<part>; a part found outside its "
                                           "zone -> the 'outside zone' defect. Both become REJECT only through the "
                                           "frame vote.", text_color=DIM, anchor="w", justify="left", wraplength=960)
        self.msg.pack(fill="x", padx=14, pady=4)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text="Check", width=90, command=self.check).pack(side="left")
        ctk.CTkButton(row, text="Save", width=90, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.save).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Use built-in bottle recipe", width=190, command=self.use_default).pack(side="left")
        ctk.CTkButton(row, text="Close", width=80, command=self.destroy).pack(side="right")
        self.after(100, self.lift)

    def recipe(self) -> dict:
        def num(e, what):
            t = e.get().strip()
            if not t:
                return None
            try:
                return float(t)
            except ValueError:
                raise ValueError(f"{what}: '{t}' is not a number") from None
        parts = []
        for i, w in enumerate(self.rows, 1):
            name = w["name"].get().strip()
            if not name:
                continue
            p = {"name": name, "required": bool(w["required"].get())}
            for k, label in (("s", "search"), ("z", "zone")):
                a, b = num(w[k + "0"], f"part {i} {label} top"), num(w[k + "1"], f"part {i} {label} bottom")
                if (a is None) != (b is None):
                    raise ValueError(f"part {i} ({name}): give both {label} top and bottom, or neither")
                if a is not None:
                    p[label] = [a, b]
            if w["misplaced"].get().strip():
                p["misplaced"] = w["misplaced"].get().strip()
            parts.append(p)
        return {"anchor": self.anchor.get().strip(), "parts": parts}

    def problems(self, r) -> list:
        out = self.DEC.recipe_problems(r)
        if self.classes:
            missing = [c for c in self.DEC.recipe_classes(r) if c not in self.classes]
            if missing:
                out.append(f"the detector has no class {', '.join(missing)} (it knows {', '.join(self.classes)})")
        return out

    def check(self) -> bool:
        try:
            r = self.recipe()
        except ValueError as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return False
        p = self.problems(r)
        self.msg.configure(text=("Recipe problems:\n- " + "\n- ".join(p)) if p else
                           f"OK: anchor {r['anchor']}, {len(r['parts'])} part(s), "
                           f"{sum(1 for x in r['parts'] if x['required'])} required.",
                           text_color=BAD if p else GOOD)
        return not p

    def save(self) -> bool:
        if self.app.tab_production.running:
            self.msg.configure(text="Stop the line before changing the recipe.", text_color=BAD)
            return False
        if not self.check():
            return False
        cfg = D.load_config()
        cfg["inspection"] = self.recipe()
        D.save_config(cfg)
        from line import applog
        from line import production_store
        applog.log("app", "inspection recipe saved", project=D.PROJECT,
                   recipe_id=production_store.recipe_id(cfg["inspection"]))
        self.msg.configure(text=self.msg.cget("text") + "  Saved to config.json.", text_color=GOOD)
        if self.on_saved:
            self.on_saved()
        return True

    def use_default(self):
        if self.app.tab_production.running:
            return self.msg.configure(text="Stop the line before changing the recipe.", text_color=BAD)
        if not messagebox.askyesno("Recipe", "Remove the project recipe and use the built-in bottle / cap / label rule?",
                                   parent=self):
            return
        cfg = D.load_config()
        cfg.pop("inspection", None)
        D.save_config(cfg)
        self.msg.configure(text="Project recipe removed: the built-in bottle rule is used.", text_color=GOOD)
        if self.on_saved:
            self.on_saved()


# =============================================================================== simulation check
class SimulationCheckDialog(ctk.CTkToplevel):
    """"Is it the ladder or is it the code?"  Two read-only checks, no PLC, no cameras, no hardware:

      1. LADDER: the saved ISPSoft program is run in a scan simulator (plc.ladder_sim) against the software's
         PASS / REJECT handshake, timing, bottle spacing and safety expectations.
      2. SOFTWARE: every module self-test (selfcheck.py) in its own process, against a fake PLC that emulates the
         decoded ladder, fake cameras and fake models.

    If 1 FAILs the ladder (or the T0/T1 entered in the app) is wrong; if 2 FAILs the code is wrong. Both can pass
    and the machine still fail: neither is a hardware test."""

    COLOUR = {"PASS": GOOD, "LIMIT": WARN, "WARN": WARN, "FAIL": BAD, "SKIP": DIM}

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Simulation check - ENGINEER ONLY (commissioning aid; to be removed once the system is proven)")
        self.geometry("1000x720")
        self.transient(app)
        f = box(self, "LADDER FILE (read only)", fill="x", padx=12, pady=(12, 6))
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))
        from plc import ladder_check as LC
        self.path = ctk.CTkEntry(row, width=640)
        self.path.insert(0, str(LC.DEFAULT_ISP))
        self.path.pack(side="left")
        ctk.CTkButton(row, text="Browse...", width=90, command=self.browse).pack(side="left", padx=6)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        self.b_lad = ctk.CTkButton(row, text="1  Check the LADDER (simulation)", width=280, fg_color=ACC,
                                   text_color=ACC_T, hover_color=ACC_H, command=self.check_ladder)
        self.b_lad.pack(side="left")
        self.b_sw = ctk.CTkButton(row, text="2  Check the SOFTWARE (self-tests, ~1 min)", width=320,
                                  command=lambda: self.check_software(False))
        self.b_sw.pack(side="left", padx=8)
        self.b_full = ctk.CTkButton(row, text="2b  Full (incl. GUI, models)", width=200,
                                    command=lambda: self.check_software(True))
        self.b_full.pack(side="left")
        self.b_plc = ctk.CTkButton(row, text="3  Read the connected PLC now", width=240, command=self.read_plc)
        self.b_plc.pack(side="left", padx=8)
        self.verdict = ctk.CTkLabel(self, text="", font=theme.H2, anchor="w", justify="left", wraplength=960)
        self.verdict.pack(fill="x", padx=14, pady=(8, 2))
        self.out = ctk.CTkTextbox(self, font=MONO, wrap="word")
        self.out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        for k, c in self.COLOUR.items():
            self.out._textbox.tag_config(k, foreground=c)
        self.out._textbox.tag_config("DIM", foreground=DIM)
        self.out.insert("end", "Press 1 to test the ladder file against what the software needs, 2 to test the code.\n"
                               "Both are simulations: they find logic and timing mistakes, they do not replace the "
                               "physical tests.")
        self.out.configure(state="disabled")
        self.after(100, self.lift)

    def browse(self):
        p = filedialog.askopenfilename(parent=self, title="ISPSoft project", filetypes=[("ISPSoft", "*.isp"), ("All", "*.*")])
        if p:
            self.path.delete(0, "end")
            self.path.insert(0, p)

    def _put(self, lines):
        """lines: [(text, tag)]"""
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        for t, tag in lines:
            self.out._textbox.insert("end", t + "\n", tag)
        self.out.configure(state="disabled")

    def check_ladder(self):
        from plc import ladder_sim as LS
        cfg = dict(self.app.settings)
        try:
            v, c, res, nets = LS.simulate(self.path.get().strip(), cfg)
        except Exception as e:                                         # noqa: BLE001 - shown
            self.verdict.configure(text=f"Cannot read the ladder: {e}", text_color=BAD)
            return
        bad = c.get("FAIL", 0)
        self.verdict.configure(text=v + "   " + "  ".join(f"{k} {n}" for k, n in c.items() if n),
                               text_color=BAD if bad else (WARN if c.get("WARN") or c.get("LIMIT") else GOOD))
        lines = [(f"settings.json used: T0 {cfg.get('plc_t0_s')} s, T1 {cfg.get('plc_t1_s')} s, E-stop input "
                  f"{cfg.get('estop_device') or 'none'}", "DIM"),
                 ("networks: " + " | ".join(n.text() for n in nets), "DIM"), ("", "DIM")]
        for r in res:
            lines.append((f"{r.status:<5} {r.id}  {r.title}", r.status))
            lines.append((f"        {r.detail}", "DIM"))
            if r.status in ("FAIL", "WARN", "LIMIT") and r.impact:
                lines.append((f"        why it matters: {r.impact}", "DIM"))
        lines += [("", "DIM"), ("What to do: docs/hardware/PLC_LADDER_REQUIREMENTS.md lists the rung for every FAIL / WARN.",
                                 "DIM")]
        self._put(lines)

    def read_plc(self):
        """Read-only snapshot of whatever PLC the app is connected to (SIMULATOR or the REAL one): the same bits the
        ladder simulation reasons about, so the two can be compared by eye. Writes nothing."""
        svc = self.app.plc
        snap = svc.snapshot()
        mode = "SIMULATOR" if svc.simulator_mode else "REAL PLC"
        if snap is None:
            self.verdict.configure(text=f"{mode}: not connected (Simulation testing page -> Connect)", text_color=WARN)
            return self._put([("No live PLC data. Connect on the Simulation testing page first; this button only reads.", "DIM")])
        self.verdict.configure(text=f"{mode} {svc.client.transport.description}   PLC {'RUN' if snap['plc_run'] else 'STOP'}   "
                                    f"data {snap['age_s']:.1f} s old", text_color=GOOD if snap["plc_run"] else WARN)
        lines = []
        for k, title in (("inputs", "INPUTS X"), ("internal", "INTERNAL M"), ("outputs", "OUTPUTS Y"),
                         ("timers", "TIMER VALUES T (x 0.1 s)"), ("counters", "COUNTERS C")):
            lines.append((title, "DIM"))
            lines.append(("   " + "   ".join(f"{n}={v}" for n, v in snap[k].items()), "PASS"))
        lines += [("", "DIM"), ("Compare with the ladder simulation: e.g. with the real ladder, a REJECT should keep M1 ON while T0 counts "
                               "up to its preset, then Y0 ON for T1.", "DIM")]
        self._put(lines)

    def check_software(self, full):
        from tools import selfcheck
        for b in (self.b_lad, self.b_sw, self.b_full, self.b_plc):
            b.configure(state="disabled")
        self.verdict.configure(text="running self-tests...", text_color=WARN)
        done = []

        def progress(label, r):
            done.append((label, r))
            self.app.post(lambda: self._show_sw(done, running=True))

        def work():
            return selfcheck.run_all(full, progress)

        def finish(res):
            self._show_sw(done, running=False)
            for b in (self.b_lad, self.b_sw, self.b_full, self.b_plc):
                b.configure(state="normal")
        self.app.run_bg(work, finish)

    def _show_sw(self, done, running):
        bad = [d for d in done if not d[1][1]]
        self.verdict.configure(
            text=(f"{len(done) - len(bad)}/{len(done)} passed" + (f", {len(bad)} FAILED: the code has a problem"
                                                                if bad else ("" if running else ": the code is consistent with itself"))
                  + (" (running...)" if running else "")), text_color=BAD if bad else (WARN if running else GOOD))
        lines = []
        for label, (_, ok, msg, sec, why) in done:
            lines.append((f"{'PASS' if ok else 'FAIL'}  {label}  ({sec:.1f} s)", "PASS" if ok else "FAIL"))
            lines.append((f"        {why}", "DIM"))
            if not ok:
                lines.append((f"        {msg}", "FAIL"))
        lines += [("", "DIM"), ("Software-only checks with fakes; not a hardware test.", "DIM")]
        self._put(lines)


# =============================================================================== database
class DatabaseTab:
    """The production database, readable: a table of what was stored (bottles / alarms / runs), a plain-words
    explanation of every column, the file location, and export (CSV for Excel, printable HTML report)."""

    RANGES = ("Today", "Last 7 days", "Last 30 days", "All time", "Shift A today", "Shift B today", "Shift C today")

    def __init__(self, app, parent):
        self.app = app
        self.cols: list = []
        self.rows: list = []
        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 6))
        r1 = ctk.CTkFrame(bar, fg_color="transparent")
        r1.pack(fill="x", padx=8, pady=(8, 2))
        ctk.CTkLabel(r1, text="Table").pack(side="left", padx=(4, 4))
        self.table = ctk.CTkSegmentedButton(r1, values=["Bottles", "Alarms", "Runs"], command=lambda _=None: self.refresh())
        self.table.set("Bottles")
        self.table.pack(side="left")
        ctk.CTkLabel(r1, text="Range").pack(side="left", padx=(14, 4))
        self.range = ctk.CTkOptionMenu(r1, values=list(self.RANGES), width=140, command=lambda _=None: self.refresh())
        self.range.pack(side="left")
        ctk.CTkLabel(r1, text="Result").pack(side="left", padx=(14, 4))
        self.result = ctk.CTkOptionMenu(r1, values=["All", "GOOD", "DEFECT", "FAULT"], width=100,
                                        command=lambda _=None: self.refresh())
        self.result.pack(side="left")
        self.q = ctk.CTkEntry(r1, width=200, placeholder_text="search (e.g. missing cap, 000012)")
        self.q.pack(side="left", padx=(14, 4))
        self.q.bind("<Return>", lambda e: self.refresh())
        ctk.CTkButton(r1, text="Search", width=70, command=self.refresh).pack(side="left")
        r2 = ctk.CTkFrame(bar, fg_color="transparent")
        r2.pack(fill="x", padx=8, pady=(2, 8))
        ctk.CTkButton(r2, text="Export CSV (Excel)", width=150, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.export_csv).pack(side="left", padx=(4, 6))
        ctk.CTkButton(r2, text="Production report (print / PDF)", width=230, command=self.export_report).pack(side="left", padx=6)
        ctk.CTkButton(r2, text="Open evidence picture", width=170, command=self.open_evidence).pack(side="left", padx=6)
        ctk.CTkButton(r2, text="Open database folder", width=170, command=self.open_folder).pack(side="left", padx=6)
        self.info = ctk.CTkLabel(parent, text="", font=MONO, text_color=DIM, anchor="w", justify="left")
        self.info.pack(fill="x", padx=6, pady=(0, 4))

        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3, uniform="d", minsize=0)
        body.grid_columnconfigure(1, weight=2, uniform="d", minsize=0)
        body.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(body, fg_color=PANEL, border_width=1, border_color=LINE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(left, text="ROWS (click a column title to sort; scroll with the bars)", font=theme.CAPS,
                     text_color=DIM).pack(anchor="w", padx=12, pady=(8, 2))
        wrap = tk.Frame(left, bg=PANEL, width=100, height=100)
        wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        wrap.grid_propagate(False)                       # the wide table must scroll, never push the explanation away
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Prod.Treeview", background=FIELD, foreground=INK, fieldbackground=FIELD, rowheight=24,
                        bordercolor=LINE, font=("Consolas", 12))
        style.configure("Prod.Treeview.Heading", background=PANEL_2, foreground=INK, font=("Segoe UI", 12, "bold"),
                        relief="flat")
        style.map("Prod.Treeview", background=[("selected", ACC_SOFT)], foreground=[("selected", INK)])
        self.tree = ttk.Treeview(wrap, style="Prod.Treeview", show="headings", selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs = ctk.CTkScrollbar(wrap, orientation="vertical", command=self.tree.yview)
        vs.grid(row=0, column=1, sticky="ns")
        hs = ctk.CTkScrollbar(wrap, orientation="horizontal", command=self.tree.xview)
        hs.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.tag_configure("GOOD", foreground=GOOD)
        self.tree.tag_configure("DEFECT", foreground=BAD)
        self.tree.tag_configure("FAULT", foreground=WARN)
        self._sort = (None, False)

        right = ctk.CTkFrame(body, fg_color=PANEL, border_width=1, border_color=LINE)
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(right, text="WHAT THE COLUMNS MEAN, AND HOW THE DATABASE WORKS", font=theme.CAPS,
                     text_color=DIM).pack(anchor="w", padx=12, pady=(8, 2))
        self.explain = ctk.CTkTextbox(right, font=("Segoe UI", 13), wrap="word")
        self.explain.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ------------------------------------------------------------------ data
    def span(self):
        r = self.range.get()
        now = time.time()
        today = time.strftime("%Y-%m-%d")
        t0 = time.mktime(time.strptime(today, "%Y-%m-%d"))
        if r == "Today":
            return t0, t0 + 86400
        if r == "Last 7 days":
            return now - 7 * 86400, now + 60
        if r == "Last 30 days":
            return now - 30 * 86400, now + 60
        if r == "All time":
            return None
        from line import production_store
        name = r.split()[1]
        sh = next((x for x in production_store.DEFAULT_SHIFTS if x[0] == name), None)
        return production_store.shift_span(today, sh)

    def key(self):
        return {"Bottles": "inspections", "Alarms": "alarms", "Runs": "runs"}[self.table.get()]

    def refresh(self):
        from line import production_export as PE
        st = self.app.store()
        key = self.key()
        self.cols, self.rows = PE.rows_for(st, key, self.span(), self.result.get() if key == "inspections" else "All",
                                           self.q.get(), limit=5000)
        self._draw()
        i = PE.info(st)
        self.info.configure(text=f"file {i['path']}   {i['size_kb']} KB   bottles {i['rows']['inspections']}   "
                                 f"alarms {i['rows']['alarms']}   runs {i['rows']['runs']}   "
                                 f"first {i['first'] or '-'}   last {i['last'] or '-'}   showing {len(self.rows)}")
        ex = [f"- {k}: {v}" for k, v in PE.EXPLAIN[key]]
        self.explain.configure(state="normal")
        self.explain.delete("1.0", "end")
        self.explain.insert("end", f"TABLE: {self.table.get().upper()}\n" + "\n".join(ex) + "\n\n" + PE.ABOUT)
        self.explain.configure(state="disabled")

    def _draw(self):
        tv = self.tree
        tv.delete(*tv.get_children())
        tv["columns"] = self.cols
        for c in self.cols:
            tv.heading(c, text=c.upper(), command=lambda c=c: self._sort_by(c))
            tv.column(c, width=max(90, min(300, 9 * max([int(len(c) * 1.4)] + [len(str(r[self.cols.index(c)])) for r in self.rows[:60]]))),
                      anchor="w", stretch=False)
        for r in self.rows:
            tag = r[self.cols.index("result")] if "result" in self.cols else (
                "FAULT" if self.cols and self.cols[0] == "time" and "severity" in self.cols else "")
            tv.insert("", "end", values=r, tags=(tag,) if tag in ("GOOD", "DEFECT", "FAULT") else ())

    def _sort_by(self, c):
        i = self.cols.index(c)
        rev = self._sort == (c, False)
        self.rows.sort(key=lambda r: (str(r[i]).isdigit() is False, float(r[i]) if str(r[i]).replace(".", "", 1).isdigit()
                                      else str(r[i])), reverse=rev)
        self._sort = (c, rev)
        self._draw()

    # ------------------------------------------------------------------ export
    def export_csv(self):
        from line import production_export as PE
        key = self.key()
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"production_{key}_{time.strftime('%Y%m%d')}.csv",
                                           filetypes=[("CSV (Excel)", "*.csv")])
        if not out:
            return
        n = PE.export_csv(self.app.store(), key, out, self.span(), self.result.get() if key == "inspections" else "All",
                          self.q.get())
        messagebox.showinfo("Export", f"{n} row(s) written to\n{out}\n\nOpen it with Excel.")

    def export_report(self):
        from line import production_export as PE
        span = self.span() or (0.0, time.time() + 60)
        out = filedialog.asksaveasfilename(defaultextension=".html", initialfile=f"production_report_{time.strftime('%Y%m%d')}.html",
                                           filetypes=[("HTML report", "*.html")])
        if not out:
            return
        PE.export_report(self.app.store(), out, span, f"Production report - {D.project_title(D.PROJECT)}",
                         self.range.get())
        try:
            os.startfile(out)                                              # noqa: S606 - local file
        except (AttributeError, OSError):
            pass
        messagebox.showinfo("Report", f"Saved to\n{out}\n\nIn the browser: Print -> Save as PDF.")

    def open_evidence(self):
        sel = self.tree.selection()
        if not sel or "evidence" not in self.cols:
            return messagebox.showinfo("Evidence", "Select a bottle row first.")
        ev = self.tree.item(sel[0], "values")[self.cols.index("evidence")]
        if not ev:
            return messagebox.showinfo("Evidence", "No picture was kept for this bottle (see Settings -> evidence images kept).")
        p = self.app.store().folder / ev
        if p.exists():
            try:
                os.startfile(str(p))                                       # noqa: S606
            except (AttributeError, OSError) as e:
                messagebox.showinfo("Evidence", f"{p}\n({e})")
        else:
            messagebox.showwarning("Evidence", f"{p} is missing.")

    def open_folder(self):
        try:
            os.startfile(str(self.app.store().folder))                     # noqa: S606
        except (AttributeError, OSError) as e:
            messagebox.showinfo("Folder", f"{self.app.store().folder}\n({e})")


# =============================================================================== validation form
class ValidateDialog(ctk.CTkToplevel):
    """The form behind Models -> Validate: what was shown to the REAL camera and what the model said, plus (for a
    classifier) the background-shortcut check. model_registry.validate refuses the model unless the numbers are
    within the gates in settings.json "activation_gates" (model_checks.DEFAULT_GATES)."""

    FIELDS = (("good_n", "Real GOOD bottles shown"), ("good_called_defective", "...of them called defective"),
              ("defective_n", "Real DEFECTIVE bottles shown"), ("defective_passed", "...of them passed as good"))

    def __init__(self, app, reg, entry, on_done=None):
        super().__init__(app)
        from registry import model_checks as MCH
        self.app, self.reg, self.e, self.on_done, self.MCH = app, reg, entry, on_done, MCH
        self.shortcut = None
        g = MCH.gates(app.settings)
        self.title(f"Validate {entry['name']}")
        self.geometry("720x560")
        self.transient(app)
        f = box(self, "WHAT THE MODEL DID ON THE REAL CAMERA (count bottles, not frames)", fill="x", padx=12, pady=(12, 6))
        self.ent = {}
        for k, label in self.FIELDS:
            row = ctk.CTkFrame(f, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=label, width=260, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, width=90)
            e.pack(side="left")
            self.ent[k] = e
        ctk.CTkLabel(f, text=f"Limits: good called defective <= {g['max_good_called_defective_pct']:g} %, defective "
                             f"passed <= {g['max_defective_passed_pct']:g} %, at least {g['min_real_good']} good and "
                             f"{g['min_real_defective']} defective bottles (settings.json activation_gates).",
                     text_color=DIM, wraplength=660, justify="left").pack(anchor="w", padx=12, pady=(4, 8))
        ctk.CTkLabel(f, text="How the bottles were shown (camera, enclosure, lighting, which defects):",
                     anchor="w").pack(anchor="w", padx=12)
        self.note = ctk.CTkEntry(f, width=660)
        self.note.pack(padx=12, pady=(2, 10))
        if entry["kind"] == "classification":
            f2 = box(self, "BACKGROUND-SHORTCUT CHECK (classifier only)", fill="x", padx=12, pady=6)
            row = ctk.CTkFrame(f2, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(0, 10))
            self.b_sc = ctk.CTkButton(row, text="Run the check", width=140, command=self.run_shortcut)
            self.b_sc.pack(side="left")
            self.sc_lbl = ctk.CTkLabel(row, text="capped white-background bottles must NOT be called missing_cap",
                                       text_color=DIM)
            self.sc_lbl.pack(side="left", padx=10)
        self.msg = ctk.CTkLabel(self, text="", wraplength=680, justify="left", anchor="w")
        self.msg.pack(fill="x", padx=14, pady=6)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text="VALIDATE", width=120, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.submit).pack(side="left")
        ctk.CTkButton(row, text="Close", width=80, command=self.destroy).pack(side="right")
        self.after(100, self.lift)

    def run_shortcut(self):
        self.b_sc.configure(state="disabled", text="running...")
        stamp = self.e["id"]

        def done(r):
            self.shortcut = r
            self.b_sc.configure(state="normal", text="Run again")
            if r.get("skipped"):
                self.sc_lbl.configure(text=f"not runnable here: {r['skipped']}", text_color=WARN)
            else:
                self.sc_lbl.configure(text=f"{r['fired']}/{r['n']} capped bottles called missing_cap ({r['fire_pct']} %) - "
                                           + ("PASSED" if r["passed"] else "FAILED: the model learned the background"),
                                      text_color=GOOD if r["passed"] else BAD)
        self.app.run_bg(lambda: self.MCH.shortcut_check(stamp), done)

    def numbers(self) -> dict:
        out = {}
        for k, label in self.FIELDS:
            try:
                out[k] = int(self.ent[k].get())
            except ValueError:
                raise ValueError(f"'{label}' must be a whole number") from None
        return out

    def submit(self):
        try:
            real = self.numbers()
            self.reg.validate(self.e["name"], self.note.get(), real=real, shortcut=self.shortcut)
        except (ValueError, MR.RegistryError) as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return
        self.msg.configure(text="VALIDATED. Next: Approve, then ACTIVATE (with the line stopped).", text_color=GOOD)
        if self.on_done:
            self.on_done()


# ---------------------------------------------------------------------------------- camera ROI + arrival timing
def _frame_of(app, src):
    cam = app.cams.get(src)
    f = cam.latest_frame() if cam is not None else None
    return None if f is None else f


class RoiDialog(ctk.CTkToplevel):
    """Draw, per camera, the region where a bottle counts. Saved as fractions of the frame (so a different capture
    size keeps the same region) in settings.json "camera_roi". A bottle whose box centre is outside it is "no bottle"
    for that camera. Needs the camera running (CAMERA TEST)."""

    W, H = 800, 450

    def __init__(self, app, srcs, on_saved=None):
        super().__init__(app)
        from PIL import ImageTk
        self._ImageTk = ImageTk
        self.app, self.srcs, self.on_saved = app, list(srcs), on_saved
        self.title("Region of interest per camera")
        self.geometry(f"{self.W + 40}x{self.H + 190}")
        self.transient(app)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(12, 4))
        ctk.CTkLabel(row, text="Camera").pack(side="left")
        self.pick = ctk.CTkOptionMenu(row, values=[n for _, n in self.srcs], width=300, command=lambda _=None: self._load())
        self.pick.pack(side="left", padx=8)
        self.info = ctk.CTkLabel(row, text="", text_color=DIM)
        self.info.pack(side="left", padx=8)
        self.cv = tk.Canvas(self, width=self.W, height=self.H, bg="#1a1d23", highlightthickness=0, cursor="crosshair")
        self.cv.pack(padx=12, pady=4)
        self.cv.bind("<ButtonPress-1>", self._down)
        self.cv.bind("<B1-Motion>", self._move)
        self.cv.bind("<ButtonRelease-1>", self._up)
        ctk.CTkLabel(self, text="Drag a rectangle around where the bottle passes in front of this camera. Only bottles "
                                "whose centre is inside it are inspected.", text_color=DIM, wraplength=self.W).pack(padx=12)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=8)
        ctk.CTkButton(row, text="Save ROI for this camera", fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.save).pack(side="left")
        ctk.CTkButton(row, text="Clear (whole frame)", command=self.clear).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Close", command=self.destroy).pack(side="right")
        self.msg = ctk.CTkLabel(self, text="", text_color=DIM)
        self.msg.pack(padx=12, anchor="w")
        self.rect = None                          # (x0, y0, x1, y1) canvas pixels
        self._img = None
        self._scale = (1.0, 0, 0)                 # k, x offset, y offset of the picture inside the canvas
        self._drag = None
        self._load()
        self._tick()

    def _src(self):
        names = {n: s for s, n in self.srcs}
        return names.get(self.pick.get())

    def _load(self):
        """Show the saved ROI of the chosen camera."""
        self.rect = None
        r = (self.app.settings.get("camera_roi") or {}).get(str(self._src()))
        self._saved = r
        self._place_saved = r is not None

    def _tick(self):
        if not self.winfo_exists():
            return
        f = _frame_of(self.app, self._src())
        self.cv.delete("all")
        if f is None:
            self.cv.create_text(self.W // 2, self.H // 2, fill="white",
                                text="no picture: press CAMERA TEST on the Production tab first")
        else:
            import cv2
            from PIL import Image
            h, w = f.image.shape[:2]
            k = min(self.W / w, self.H / h)
            nw, nh = int(w * k), int(h * k)
            ox, oy = (self.W - nw) // 2, (self.H - nh) // 2
            self._scale = (k, ox, oy, nw, nh)
            small = cv2.resize(f.image, (nw, nh), interpolation=cv2.INTER_AREA)
            self._img = self._ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.cv.create_image(ox, oy, anchor="nw", image=self._img)
            if self._place_saved and self._saved:
                x, y, rw, rh = self._saved
                self.rect = (ox + x * nw, oy + y * nh, ox + (x + rw) * nw, oy + (y + rh) * nh)
                self._place_saved = False
            self.info.configure(text=f"{w}x{h}")
        if self.rect:
            self.cv.create_rectangle(*self.rect, outline="#e6c828", width=2)
        self.after(120, self._tick)

    def _down(self, e):
        self._drag = (e.x, e.y)
        self.rect = (e.x, e.y, e.x, e.y)

    def _move(self, e):
        if self._drag:
            self.rect = (min(self._drag[0], e.x), min(self._drag[1], e.y), max(self._drag[0], e.x), max(self._drag[1], e.y))

    def _up(self, e):
        self._drag = None

    def save(self):
        if not self.rect or len(self._scale) < 5:
            self.msg.configure(text="draw a rectangle on the picture first", text_color=BAD)
            return
        k, ox, oy, nw, nh = self._scale
        x0, y0 = max(0.0, (self.rect[0] - ox) / nw), max(0.0, (self.rect[1] - oy) / nh)
        x1, y1 = min(1.0, (self.rect[2] - ox) / nw), min(1.0, (self.rect[3] - oy) / nh)
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            self.msg.configure(text="the region is too small (under 5% of the frame)", text_color=BAD)
            return
        rois = dict(self.app.settings.get("camera_roi") or {})
        rois[str(self._src())] = [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]
        self.app.settings["camera_roi"] = rois
        D.save_settings(self.app.settings)
        self.msg.configure(text=f"saved {rois[str(self._src())]} (fractions of the frame). Applies at the next START.",
                           text_color=GOOD)
        if self.on_saved:
            self.on_saved()

    def clear(self):
        rois = dict(self.app.settings.get("camera_roi") or {})
        rois.pop(str(self._src()), None)
        self.app.settings["camera_roi"] = rois
        D.save_settings(self.app.settings)
        self.rect = None
        self.msg.configure(text="ROI removed: the whole frame counts for this camera", text_color=DIM)


class ArrivalTimingDialog(ctk.CTkToplevel):
    """Measure, on real bottles, when each one shows up in a camera after the photo-eye (X0) sees it, and for how long
    it stays in the ROI. From that it proposes capture_delay_s and inspect_window_s. Resolution is the detector's frame
    rate (about 0.1 s) plus the PLC poll (0.05 s): good enough to set a window of a second or more.

    Needs: PLC connected, CAMERA TEST running, the line stopped. It only READS the PLC (X0) and the camera."""

    GAP_S = 0.5               # a bottle that has been absent this long has left the view
    GIVE_UP_S = 25.0          # X0 with no bottle ever seen in the ROI within this: dropped

    def __init__(self, app, srcs, on_saved=None):
        super().__init__(app)
        import threading
        self.app, self.srcs, self.on_saved = app, list(srcs), on_saved
        self.title("Arrival timing: photo-eye to camera")
        self.geometry("720x640")
        self.transient(app)
        self.rows: list = []                  # (arrive_s, leave_s)
        self.state = "starting"
        self.lock = threading.Lock()
        self._x0: list = []
        self._stop = threading.Event()
        self._prev_watch = app.plc.watch_x0
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(12, 4))
        ctk.CTkLabel(row, text="Camera").pack(side="left")
        self.pick = ctk.CTkOptionMenu(row, values=[n for _, n in self.srcs], width=300)
        self.pick.pack(side="left", padx=8)
        ctk.CTkButton(row, text="Reset", width=70, command=self.reset).pack(side="left")
        ctk.CTkLabel(self, text="Run bottles on the conveyor one at a time (at least 5). Each X0 pulse starts a stopwatch; "
                                "it stops when the bottle appears inside this camera's ROI and again when it leaves.",
                     text_color=DIM, wraplength=680, justify="left").pack(padx=12, anchor="w")
        self.status = ctk.CTkLabel(self, text="", anchor="w", justify="left")
        self.status.pack(fill="x", padx=12, pady=4)
        self.out = ctk.CTkTextbox(self, font=("Consolas", 13), height=330)
        self.out.pack(fill="both", expand=True, padx=12, pady=4)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=8)
        self.btn = ctk.CTkButton(row, text="Apply to settings", fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                                 state="disabled", command=self.apply)
        self.btn.pack(side="left")
        ctk.CTkButton(row, text="Close", command=self.close).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)
        try:
            self.det = app.tab_live.build_detector()
        except Exception as e:                                  # noqa: BLE001 - shown
            self.det = None
            self.state = f"detector not available: {e}"
        app.plc.watch_x0 = True
        app.plc.add_listener(self._on_event)
        if self.det is not None:
            threading.Thread(target=self._work, name="arrival-timing", daemon=True).start()
        self._tick()

    def close(self):
        self._stop.set()
        self.app.plc.watch_x0 = self._prev_watch
        try:
            self.app.plc._listeners.remove(self._on_event)
        except (ValueError, AttributeError):
            pass
        self.destroy()

    def reset(self):
        with self.lock:
            self.rows.clear()

    def _src(self):
        return {n: s for s, n in self.srcs}.get(self.pick.get())

    def _on_event(self, ev):
        if ev.device == "X0" and ev.ack == "ON":
            with self.lock:
                self._x0.append(ev.ts)

    def _work(self):
        from line import decision as DEC
        seen, rec = {}, None                    # rec: {"t0", "arrive", "last"}
        while not self._stop.is_set():
            src = self._src()
            cam = self.app.cams.get(src)
            f = cam.latest_frame() if cam is not None else None
            with self.lock:
                new_x0, self._x0 = self._x0, []
            for t0 in new_x0:
                if rec is not None and rec["arrive"] is not None:
                    self._close(rec)
                rec = {"t0": t0, "arrive": None, "last": None}
            if f is None or seen.get(src) == f.seq:
                self._stop.wait(0.02)
                continue
            seen[src] = f.seq
            self.state = "measuring"
            try:
                det = self.det.detect(f.image, camera_id=str(src), frame_seq=f.seq, frame_ts=f.ts)
                roi = (self.app.settings.get("camera_roi") or {}).get(str(src))
                if roi:
                    W, H = det.frame_wh
                    det = det._replace(detections=tuple(
                        d for d in det.detections
                        if roi[0] * W <= (d.x1 + d.x2) / 2 <= (roi[0] + roi[2]) * W
                        and roi[1] * H <= (d.y1 + d.y2) / 2 <= (roi[1] + roi[3]) * H))
                present = bool(DEC.gate_detections(det, {**DEC.RULES, **(self.app.settings.get("decision_rules") or {})}))
            except Exception as e:                              # noqa: BLE001
                self.state = f"detector error: {type(e).__name__}: {e}"
                self._stop.wait(0.5)
                continue
            if rec is None or f.ts < rec["t0"]:
                continue
            dt = f.ts - rec["t0"]
            if present:
                if rec["arrive"] is None:
                    rec["arrive"] = dt
                rec["last"] = dt
            elif rec["arrive"] is not None and dt - rec["last"] > self.GAP_S:
                self._close(rec)
                rec = None
            elif rec["arrive"] is None and dt > self.GIVE_UP_S:
                rec = None

    def _close(self, rec):
        with self.lock:
            self.rows.append((rec["arrive"], rec["last"]))

    def suggestion(self):
        with self.lock:
            rows = list(self.rows)
        if len(rows) < 3:
            return None
        arr = [a for a, _ in rows]
        lea = [b for _, b in rows]
        delay = max(arr) + 0.05                              # the latest bottle is fully in view
        window = min(lea) - delay - 0.05                     # ...and the earliest leaver is still there
        return delay, max(0.3, min(3.0, window)), window

    def _tick(self):
        if not self.winfo_exists():
            return
        with self.lock:
            rows = list(self.rows)
        lines = [f"{'#':>2}  arrives (s after X0)  leaves (s)  visible (s)"]
        for i, (a, b) in enumerate(rows, 1):
            lines.append(f"{i:>2}  {a:>10.2f}  {b:>16.2f}  {b - a:>9.2f}")
        sg = self.suggestion()
        if sg:
            d, w, raw = sg
            lines += ["", f"arrival  min {min(a for a, _ in rows):.2f}  max {max(a for a, _ in rows):.2f} s",
                      f"suggested capture_delay_s = {d:.2f}", f"suggested inspect_window_s = {w:.2f}"
                      + ("" if raw >= 0.3 else "   (bottles are visible too briefly: widen the ROI or move the camera)")]
        else:
            lines += ["", "need at least 3 bottles for a suggestion"]
        self.out.delete("1.0", "end")
        self.out.insert("end", "\n".join(lines))
        link = "PLC connected" if self.app.plc.link_state() == "CONNECTED" else "PLC NOT connected"
        cam = "camera streaming" if _frame_of(self.app, self._src()) is not None else "no camera picture (CAMERA TEST)"
        self.status.configure(text=f"{link}   |   {cam}   |   {self.state}   |   bottles measured: {len(rows)}",
                              text_color=GOOD if self.state == "measuring" and "NOT" not in link else WARN)
        self.btn.configure(state="normal" if sg else "disabled")
        self.after(300, self._tick)

    def apply(self):
        sg = self.suggestion()
        if not sg:
            return
        self.app.settings["capture_delay_s"] = round(sg[0], 2)
        self.app.settings["inspect_window_s"] = round(sg[1], 2)
        D.save_settings(self.app.settings)
        if self.on_saved:
            self.on_saved()
        self.status.configure(text=f"saved capture_delay_s {sg[0]:.2f}, inspect_window_s {sg[1]:.2f}", text_color=GOOD)


class CameraControlsDialog(ctk.CTkToplevel):
    """Camera internal settings (focus, exposure, white balance, gain, brightness, rotation) changed ON THE RUNNING
    camera, with a live sharpness / brightness readout to judge the effect. "Apply" acts at once; "Save as default"
    stores them in settings.json "camera_controls" so every later start opens the camera the same way.

    Values are in the driver's own scale (a UVC webcam differs from another), and a webcam often ignores a property
    without any error: the read-back column shows what the driver really holds. Changing the picture changes what
    the models see: after the final values re-run the ROI (Set ROI) and, for a new setup, retrain on its images."""

    FIELDS = (("focus", "Focus", "autofocus", 0, 255), ("exposure", "Exposure", "auto_exposure", -13, 0),
              ("wb_temperature", "White balance (K)", "auto_wb", 2800, 6500), ("gain", "Gain", None, 0, 255),
              ("brightness", "Brightness", None, 0, 255))
    AUTO_ON = {"autofocus": 1, "auto_exposure": 0.75, "auto_wb": 1}          # OpenCV: 0.75 = auto, 0.25 = manual
    AUTO_OFF = {"autofocus": 0, "auto_exposure": 0.25, "auto_wb": 0}

    def __init__(self, app, srcs, on_saved=None):
        super().__init__(app)
        self.app, self.srcs, self.on_saved = app, list(srcs), on_saved
        self.title("Camera settings (live)")
        self.geometry("640x580")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(12, 4))
        ctk.CTkLabel(row, text="Camera").pack(side="left")
        self.pick = ctk.CTkOptionMenu(row, values=[n for _, n in self.srcs], width=300, command=lambda _=None: self._fill())
        self.pick.pack(side="left", padx=8)
        self.q = ctk.CTkLabel(self, text="", font=("Consolas", 13), anchor="w", justify="left")
        self.q.pack(fill="x", padx=12, pady=2)
        grid = ctk.CTkFrame(self, fg_color=PANEL, border_width=1, border_color=LINE)
        grid.pack(fill="x", padx=12, pady=6)
        self.ent, self.auto, self.rb = {}, {}, {}
        for i, (key, label, auto, lo, hi) in enumerate(self.FIELDS):
            ctk.CTkLabel(grid, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=4)
            e = ctk.CTkEntry(grid, width=80)
            e.grid(row=i, column=1, padx=6)
            self.ent[key] = e
            ctk.CTkLabel(grid, text=f"({lo} .. {hi})", text_color=DIM).grid(row=i, column=2, padx=4)
            if auto:
                v = ctk.BooleanVar(value=False)
                ctk.CTkCheckBox(grid, text="auto", variable=v, width=60).grid(row=i, column=3, padx=6)
                self.auto[auto] = (v, key)
            self.rb[key] = ctk.CTkLabel(grid, text="", font=("Consolas", 12), text_color=DIM)
            self.rb[key].grid(row=i, column=4, padx=8, sticky="w")
        ctk.CTkLabel(grid, text="Rotate").grid(row=len(self.FIELDS), column=0, sticky="w", padx=10, pady=4)
        self.rot = ctk.CTkOptionMenu(grid, values=["0", "90", "180", "270"], width=80)
        self.rot.grid(row=len(self.FIELDS), column=1, padx=6)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(row, text="Apply now", fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.apply).pack(side="left")
        ctk.CTkButton(row, text="Save as default", command=self.save).pack(side="left", padx=8)
        ctk.CTkButton(row, text="All auto", command=self.all_auto).pack(side="left")
        ctk.CTkButton(row, text="Close", command=self.destroy).pack(side="right")
        self.msg = ctk.CTkLabel(self, text="", wraplength=600, justify="left", text_color=DIM)
        self.msg.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(self, text="Tip: for inspection use manual focus (autofocus off) and a fixed exposure as short as the "
                                "light allows; turn focus until the sharpness number stops growing. A blank field leaves "
                                "that control as it is.", wraplength=600, justify="left",
                     text_color=DIM).pack(fill="x", padx=12)
        self._q_at = 0.0
        self._fill()
        self._tick()

    def _src(self):
        return {n: s for s, n in self.srcs}.get(self.pick.get())

    def _cam(self):
        return self.app.cams.get(self._src())

    def _fill(self):
        c = (self.app.settings.get("camera_controls") or {}).get(str(self._src()), {})
        for k, e in self.ent.items():
            e.delete(0, "end")
            if k in c:
                e.insert(0, str(c[k]))
        for a, (v, _) in self.auto.items():
            v.set(c.get(a) in self.AUTO_ON.values() and c.get(a) not in (0, 0.25))
        self.rot.set(str(c.get("rotate", 0)))

    def _controls(self) -> dict:
        out = {}
        for k, e in self.ent.items():
            t = e.get().strip()
            if t:
                try:
                    out[k] = float(t)
                except ValueError:
                    raise ValueError(f"{k}: {t!r} is not a number") from None
        for a, (v, key) in self.auto.items():
            if v.get():
                out[a] = self.AUTO_ON[a]
                out.pop(key, None)                  # auto on: a manual value would be ignored
            elif key in out:
                out[a] = self.AUTO_OFF[a]           # a manual value switches its auto mode off first
        if int(self.rot.get()):
            out["rotate"] = int(self.rot.get())
        return out

    def apply(self):
        cam = self._cam()
        if cam is None or not cam.alive:
            self.msg.configure(text="this camera is not running: press CAMERA TEST (Production) or Start (Live) first",
                               text_color=BAD)
            return
        try:
            c = self._controls()
        except ValueError as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return
        cam.set_controls(c)
        self.msg.configure(text="applied; reading back...", text_color=DIM)
        self.after(700, self._readback)

    def _readback(self):
        cam = self._cam()
        lc = dict(cam.live_controls) if cam is not None else {}
        for k, lab in self.rb.items():
            v = lc.get(k)
            if v is None:
                lab.configure(text="")
                continue
            same = v[1] is not None and abs(float(v[1]) - float(v[0])) < 1e-6
            lab.configure(text=f"driver: {v[1]}" + ("" if same else f"   (asked {v[0]}: ignored or rounded)"))
        self.msg.configure(text="read back from the driver: a value that differs was not accepted", text_color=DIM)

    def all_auto(self):
        for v, _ in self.auto.values():
            v.set(True)
        for e in self.ent.values():
            e.delete(0, "end")

    def save(self):
        try:
            c = self._controls()
        except ValueError as e:
            self.msg.configure(text=str(e), text_color=BAD)
            return
        allc = dict(self.app.settings.get("camera_controls") or {})
        if c:
            allc[str(self._src())] = c
        else:
            allc.pop(str(self._src()), None)
        self.app.settings["camera_controls"] = allc
        D.save_settings(self.app.settings)
        self.msg.configure(text="saved: every later start opens this camera with these values", text_color=GOOD)
        if self.on_saved:
            self.on_saved()

    def _tick(self):
        if not self.winfo_exists():
            return
        cam = self._cam()
        f = cam.latest_frame() if cam is not None else None
        if f is not None and time.monotonic() - self._q_at > 0.5:
            from vision import bench
            self._q_at = time.monotonic()
            q = bench.frame_quality(f.image)
            self.q.configure(text=f"sharpness {q['sharpness']:.0f}   brightness {q['brightness']:.0f}   "
                                  f"contrast {q['contrast']:.0f}   clipped {q['clipped']:.1f}%")
        elif f is None:
            self.q.configure(text="no picture: the camera is not running")
        self.after(300, self._tick)


class TuningDialog(ctk.CTkToplevel):
    """Sliders for the two things that decide "good or defective": the DETECTOR confidence (how sure YOLO must be that
    a box is a bottle / cap / label) and the CLASSIFIER thresholds (how high a defect score must be to count). They act
    on the running line at once and the live panel shows the numbers the models give right now, so you can see where
    a good bottle sits and where a bad one sits, and put the threshold between them.

    Rule of thumb: a good bottle is called defective -> raise that defect's threshold (or the shift); a defective bottle
    passes -> lower it. Raise a detector confidence if boxes appear on things that are not bottles. "Save as default"
    stores the values in settings.json for every later start."""

    def __init__(self, app, get_inspector):
        super().__init__(app)
        from line import decision as DEC
        self.DEC, self.app, self.get_inspector = DEC, app, get_inspector
        self.title("Live tuning: confidence and thresholds")
        self.geometry("700x780")
        s = app.settings
        rules = {**DEC.RULES["det_min_conf_by_class"], **((s.get("decision_rules") or {}).get("det_min_conf_by_class") or {})}
        self.base = {k: float(v) for k, v in (D.load_config().get("thresholds") or {}).items()}
        self.det_v = {c: ctk.DoubleVar(value=float(rules.get(c, 0.4))) for c in ("bottle", "cap", "label")}
        self.shift_v = ctk.DoubleVar(value=float(s.get("threshold_shift", 0.0)))
        self.ovr = dict(s.get("threshold_overrides") or {})
        self.defect_v = {k: ctk.DoubleVar(value=float(self.ovr.get(k, min(0.99, v)))) for k, v in self.base.items() if v <= 1.0}
        top = box(self, "DETECTOR CONFIDENCE (minimum to accept a box)", fill="x", padx=12, pady=(12, 4))
        for c, v in self.det_v.items():
            self._slider(top, c, v, 0.20, 0.95, self.push)
        mid = box(self, "CLASSIFIER: shift every defect threshold", fill="x", padx=12, pady=4)
        self._slider(mid, "shift (+ = fewer defects)", self.shift_v, -0.30, 0.45, self.push)
        low = box(self, "CLASSIFIER: one threshold per defect (score must be ABOVE it to count)", fill="x", padx=12, pady=4)
        self.rows = {}
        for k, v in self.defect_v.items():
            self.rows[k] = self._slider(low, k, v, 0.05, 0.99, lambda *_: self.push(per_defect=True))
        self.live = ctk.CTkLabel(self, text="", font=("Consolas", 12), anchor="nw", justify="left")
        self.live.pack(fill="both", expand=True, padx=12, pady=4)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=8)
        ctk.CTkButton(row, text="Save as default", fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.save).pack(side="left")
        ctk.CTkButton(row, text="Reset to project values", command=self.reset).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Close", command=self.destroy).pack(side="right")
        self.msg = ctk.CTkLabel(self, text="", text_color=DIM, wraplength=660, justify="left")
        self.msg.pack(fill="x", padx=12)
        self.per_defect_touched = bool(self.ovr)
        self.push()
        self._tick()

    @staticmethod
    def _slider(parent, name, var, lo, hi, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(row, text=name, width=190, anchor="w").pack(side="left")
        sl = ctk.CTkSlider(row, from_=lo, to=hi, variable=var, width=330, command=cmd)
        sl.pack(side="left", padx=6)
        val = ctk.CTkLabel(row, text=f"{var.get():.2f}", width=50)
        val.pack(side="left")
        var.trace_add("write", lambda *_: val.configure(text=f"{var.get():.2f}"))
        return sl

    def push(self, *_, per_defect=False):
        """Send the slider values to the running line (and remember them as this session's tuning)."""
        if per_defect:
            self.per_defect_touched = True
        ins = self.get_inspector()
        if ins is None:
            return
        ovr = {k: v.get() for k, v in self.defect_v.items()} if self.per_defect_touched else {}
        ins.tune(det_conf={c: v.get() for c, v in self.det_v.items()}, shift=self.shift_v.get(), override=ovr)

    def reset(self):
        for c, v in self.det_v.items():
            v.set(self.DEC.RULES["det_min_conf_by_class"].get(c, 0.4))
        self.shift_v.set(0.0)
        for k, v in self.defect_v.items():
            v.set(min(0.99, self.base[k]))
        self.per_defect_touched = False
        self.push()
        self.msg.configure(text="back to the thresholds the project was trained with", text_color=DIM)

    def save(self):
        s = self.app.settings
        rules = dict(s.get("decision_rules") or {})
        rules["det_min_conf_by_class"] = {c: round(v.get(), 3) for c, v in self.det_v.items()}
        s["decision_rules"] = rules
        s["threshold_shift"] = round(self.shift_v.get(), 3)
        s["threshold_overrides"] = ({k: round(v.get(), 3) for k, v in self.defect_v.items()}
                                    if self.per_defect_touched else {})
        D.save_settings(s)
        self.msg.configure(text="saved: every later START uses these values", text_color=GOOD)

    def _tick(self):
        if not self.winfo_exists():
            return
        ins = self.get_inspector()
        lines = []
        info = getattr(ins, "live_info", None) if ins is not None else None
        if not info:
            lines.append("no live data: START INSPECTION (the numbers below are what the models give right now)")
        else:
            thr = ins.thresholds()
            for cid, d in info.items():
                if time.monotonic() - d["t"] > 2.0:
                    continue
                lines.append(f"{cid}")
                best = {}
                for cls, conf in d["det"]:
                    best[cls] = max(best.get(cls, 0.0), conf)
                lines.append("  detector  " + ("  ".join(f"{c} {v:.2f} (min {self.det_v[c].get():.2f})"
                                                       for c, v in best.items() if c in self.det_v) or "nothing found"))
                hot = sorted(d["probs"].items(), key=lambda kv: -kv[1])[:6]
                lines.append("  scores    " + "  ".join(f"{k} {p:.2f}{'*' if p > thr.get(k, 1) else ''}" for k, p in hot))
            lines.append("  (* = above its threshold: counts as a defect)")
        self.live.configure(text="\n".join(lines))
        self.after(400, self._tick)
