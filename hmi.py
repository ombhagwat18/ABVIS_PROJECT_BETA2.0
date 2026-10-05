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
from tkinter import filedialog, messagebox

import customtkinter as ctk

import alarms as AL
import charts
import dataset as D
import machine_cycle as MC
import model_registry as MR
import theme
import tracking as TR
from theme import (ACC, ACC_H, ACC_SOFT, ACC_T, BAD, DIM, FIELD, GOOD, INK, LINE, MONO, OFF, PANEL, PANEL_2,
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

    FILTERS = ("All", "PASS", "REJECT", "FAULT")

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
        for k, title, col in (("total", "TOTAL", INK), ("PASS", "PASS", GOOD), ("REJECT", "REJECT", BAD),
                              ("FAULT", "FAULT", WARN), ("yield", "PASS %", INK)):
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
        import production_store
        raw = self.app.settings.get("shifts") or production_store.DEFAULT_SHIFTS
        return [tuple(x) for x in raw if len(x) == 3]

    def span(self, day):
        import production_store
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
        f = self.filter.get()
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
            line = (f"{time.strftime('%H:%M:%S', time.localtime(r['wall'])):<9}{r['inspection_id']:<8}{r['final']:<8}"
                    f"{(r['defects'] or ('-' if r['final'] == 'PASS' else d.get('reason', '')))[:25]:<26}{conf:>5}  "
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
        names = ("COMPUTER", "PLC", "CAMERAS", "MODELS", "STORAGE / LOGS", "ALARMS")
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
        import applog
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
        import applog
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
        import bench
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
        import detect
        w = Path(self.app.settings.get("detector_weights") or detect.DEFAULT_WEIGHTS)
        lvl = "HEALTHY" if cls_ok and w.exists() else "WARNING"
        try:
            import production_store
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
        if not e:
            return
        note = ctk.CTkInputDialog(title="Real-camera validation",
                                  text=f"{e['name']}\nDescribe the real-camera validation\n"
                                       f"(camera, bottles shown, results):").get_input()
        if note is not None:
            self._do(lambda: self.reg.validate(e["name"], note), "")

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
        import decision as DEC
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
        import applog
        import production_store
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
