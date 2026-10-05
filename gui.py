"""Bottle Inspection - desktop dashboard.

Label images, manage defect types, retrain, and watch the line. All the work
lives in dataset/train/infer/calibrate; this file is only the window.

Tk is not thread-safe: every widget touch happens on the main thread, and
background work (training, thumbnail loading, the camera) reports back through
a queue drained by after().
"""
from __future__ import annotations

import csv
import gc
import json
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

import alarms as AL
import annotation_studio
import applog
import bench
import charts
import dataset as D
import decision as DEC
import detect
import hmi
import infer
import machine_cycle as MC
import machine_state as MS
import production_store
import segment
import theme
import tracking as TR
import verdict as VD
from plc import (CONNECTED as PLC_CONNECTED, DEGRADED as PLC_DEGRADED, FAULT as PLC_FAULT, SERIAL_FORMATS, PLCClient,
                 PLCService, SerialTransport, TcpTransport, serial_ports)
from plc import address_map as PLC_AM

theme.apply_ctk(ctk)

# Every colour comes from theme.py (light industrial HMI by default): grey for the interface,
# green / red / amber only for PASS / REJECT / FAULT, blue only for selection.
from theme import (ACC, ACC_H, ACC_SOFT, ACC_T, BAD, BG, DIM, FAULT, GOOD, INFO, INK, LINE, MONO, MUTED, OFF,  # noqa: E402
                   PANEL, PANEL_2, RAIL, VIDEO_BG, WARN, BAD as REJECT_C, FAULT_SOFT, PASS_SOFT, REJECT_SOFT)
PER_PAGE = 60


def bgr_to_ctk(frame: np.ndarray, size=None) -> ctk.CTkImage:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if size:
        img.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)


def plc_link(cfg: dict):
    """settings.json -> (transport, station, target). 'tcp' is the ISPSoft simulator; 'serial' is the wired PLC
    (serial link not yet verified on hardware). Nothing is opened here."""
    station = int(cfg.get("plc_station", 1))
    if cfg.get("plc_mode") == "serial":
        return (SerialTransport(str(cfg.get("plc_com", "COM1")), int(cfg.get("plc_baud", 9600)),
                                str(cfg.get("plc_format", "7E1"))), station, "Delta DVP PLC (serial)")
    return (TcpTransport(str(cfg.get("plc_host", "127.0.0.1")), int(cfg.get("plc_port", 10002))), station,
            "ISPSoft DVP-SS2 simulator")


def plc_endpoint_info(port: int):
    """Who is listening on a local TCP port and which OTHER programs are connected to it (e.g. COMMGR when
    ISPSoft is online). None if psutil is not installed or the query fails. Read-only OS query."""
    try:
        import os
        import psutil
        me, listener, others = os.getpid(), None, set()

        def name(pid):
            try:
                return f"{psutil.Process(pid).name()} (pid {pid})"
            except Exception:                            # noqa: BLE001
                return f"pid {pid}"
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == "LISTEN":
                listener = name(c.pid)
            if c.raddr and c.raddr.port == port and c.pid and c.pid != me:
                others.add(name(c.pid))
        return {"listener": listener, "others": sorted(others)}
    except Exception:                                    # noqa: BLE001
        return None


def chart_panel(parent, title, w, h, note) -> ctk.CTkCanvas:
    """A titled chart canvas + caption, the chrome shared by every chart in
    AnalysisTab and TrainTab."""
    box = ctk.CTkFrame(parent, fg_color=PANEL)
    box.pack(side="left", fill="both", expand=True, padx=(0, 8))
    ctk.CTkLabel(box, text=title, text_color=DIM,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
    cv = ctk.CTkCanvas(box, width=w, height=h, bg=PANEL, highlightthickness=0)
    cv.pack(fill="both", expand=True, padx=12, pady=(0, 4))
    ctk.CTkLabel(box, text=note, text_color=DIM, font=("Segoe UI", 12),
                 wraplength=w - 10, justify="left").pack(anchor="w", padx=12, pady=(0, 10))
    return cv


class StatusLamp(ctk.CTkFrame):
    """One status-bar indicator: a coloured lamp, a caption and the current value."""

    def __init__(self, parent, caption: str):
        super().__init__(parent, fg_color=PANEL, corner_radius=6)
        self.dot = ctk.CTkLabel(self, text="●", font=("Segoe UI", 16), text_color=OFF, width=14)
        self.dot.pack(side="left", padx=(8, 4), pady=3)
        ctk.CTkLabel(self, text=caption, font=theme.CAPS, text_color=DIM).pack(side="left")
        self.val = ctk.CTkLabel(self, text="--", font=theme.SMALL, text_color=INK)
        self.val.pack(side="left", padx=(5, 8))
        self._last = None

    def show(self, colour: str, text: str):
        text = text if len(text) <= 22 else text[:21] + "…"
        if (colour, text) != self._last:           # no redraw when nothing changed: this runs twice a second
            self._last = (colour, text)
            self.dot.configure(text_color=colour)
            self.val.configure(text=text)


class ScrollHost(ctk.CTkFrame):
    """A page that scrolls up/down AND left/right when its content is bigger than the window, so a panel full of text
    can never push its buttons out of reach. `inner` is where the page builds its widgets (exactly like the plain
    frame it replaces). The scrollbars appear only when needed; the content fills the window when it fits.

    Mouse wheel: scrolls the page when the pointer is over the page itself; Shift + wheel scrolls sideways. Text boxes,
    lists and other scrollable panels keep their own wheel (App installs one handler that skips them)."""

    def __init__(self, master, min_size=(1000, 560)):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.min_w, self.min_h = min_size
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.hbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self._vset, xscrollcommand=self._hset)
        self.inner = ctk.CTkFrame(self.canvas, fg_color="transparent", corner_radius=0)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self._busy = False
        self.canvas.bind("<Configure>", self._fit)
        self.inner.bind("<Configure>", self._fit)

    def _vset(self, a, b):
        self.vbar.set(a, b)
        (self.vbar.grid_remove if float(a) <= 0.0 and float(b) >= 1.0 else
         lambda: self.vbar.grid(row=0, column=1, sticky="ns"))()

    def _hset(self, a, b):
        self.hbar.set(a, b)
        (self.hbar.grid_remove if float(a) <= 0.0 and float(b) >= 1.0 else
         lambda: self.hbar.grid(row=1, column=0, sticky="ew"))()

    def _fit(self, _e=None):
        if self._busy:
            return
        self._busy = True
        try:
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            w = max(cw, self.min_w, self.inner.winfo_reqwidth())
            h = max(ch, self.min_h, self.inner.winfo_reqheight())
            self.canvas.itemconfigure(self._win, width=w, height=h)
            self.canvas.configure(scrollregion=(0, 0, w, h))
        finally:
            self._busy = False

    def wheel(self, event):
        if event.state & 0x1:                                   # Shift = sideways
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


class NavShell(ctk.CTkFrame):
    """Left navigation rail + page area. A drop-in for the CTkTabview it replaced: add(), tab(), get() and set()
    behave the same, so every tab class and the selftest are unchanged. Pages are built once and swapped with
    grid / grid_forget, never rebuilt."""

    def __init__(self, master, groups, on_show=None):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.on_show = on_show
        self.rail = ctk.CTkFrame(self, fg_color=RAIL, corner_radius=0, width=176)
        self.rail.pack(side="left", fill="y")
        self.rail.pack_propagate(False)
        self.body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.body.pack(side="left", fill="both", expand=True)
        self.title = ctk.CTkLabel(self.body, text="", font=theme.H1, text_color=INK, anchor="w")
        self.title.pack(fill="x", padx=18, pady=(10, 4))
        self.area = ctk.CTkFrame(self.body, fg_color="transparent", corner_radius=0)
        self.area.pack(fill="both", expand=True, padx=(14, 14), pady=(0, 10))
        self.area.grid_rowconfigure(0, weight=1)
        self.area.grid_columnconfigure(0, weight=1)
        self.pages: dict[str, ctk.CTkFrame] = {}
        self.hosts: dict[str, ScrollHost] = {}
        self.buttons: dict[str, ctk.CTkButton] = {}
        self._group_of = {n: g for g, names in groups for n in names}
        self._group_frames: dict[str, ctk.CTkFrame] = {}
        self._group_labels: dict = {}
        self.groups = groups
        for g, _ in groups:
            lab = ctk.CTkLabel(self.rail, text=g, font=theme.CAPS, text_color=MUTED, anchor="w")
            lab.pack(fill="x", padx=16, pady=(14, 2))
            f = ctk.CTkFrame(self.rail, fg_color="transparent")
            f.pack(fill="x")
            self._group_frames[g] = f
            self._group_labels[g] = lab
        self.current: str | None = None
        self.visible: set | None = None              # None = every page (engineer mode)

    MIN_SIZE = {"Production": (1180, 880), "Live": (1180, 760), "Machine": (1180, 820), "Health": (1100, 760),
                "History": (1100, 700), "Models": (1100, 620), "Annotate": (1180, 760), "Database": (1100, 700)}

    def add(self, name: str) -> ctk.CTkFrame:
        host = ScrollHost(self.area, self.MIN_SIZE.get(name, (1000, 560)))
        page = host.inner
        self.hosts[name] = host
        self.pages[name] = page
        holder = self._group_frames.get(self._group_of.get(name)) or self.rail
        b = ctk.CTkButton(holder, text="   " + name, anchor="w", height=34, corner_radius=4,
                          fg_color="transparent", hover_color=PANEL_2, text_color=DIM,
                          font=theme.BODY, border_spacing=0, command=lambda n=name: self.set(n))
        b.pack(fill="x", padx=8, pady=1)
        self.buttons[name] = b
        if self.current is None:
            self.set(name)
        return page

    def tab(self, name: str) -> ctk.CTkFrame:
        return self.pages[name]

    def show_only(self, names=None):
        """Operator mode: only these pages in the rail (None = all). Pages stay built; nothing is destroyed."""
        self.visible = set(names) if names is not None else None
        for g, lab in self._group_labels.items():
            lab.pack_forget()
            self._group_frames[g].pack_forget()
        for g, members in self.groups:
            shown = [n for n in members if n in self.buttons and (self.visible is None or n in self.visible)]
            for n in members:
                if n in self.buttons:
                    self.buttons[n].pack_forget()
            if not shown:
                continue
            self._group_labels[g].pack(fill="x", padx=16, pady=(14, 2))
            self._group_frames[g].pack(fill="x")
            for n in shown:
                self.buttons[n].pack(fill="x", padx=8, pady=1)

    def get(self) -> str:
        return self.current or ""

    def set(self, name: str):
        if name not in self.pages or name == self.current:
            return
        if self.current is not None:
            self.hosts[self.current].grid_forget()
            self.buttons[self.current].configure(fg_color="transparent", text_color=DIM)
        self.current = name
        self.hosts[name].grid(row=0, column=0, sticky="nsew")
        self.buttons[name].configure(fg_color=ACC_SOFT, text_color=INK)
        self.title.configure(text=name.upper())
        if self.on_show:
            self.on_show(name)


class App(ctk.CTk):
    def __init__(self, plc_autoconnect=None):
        super().__init__()
        self.title("Vision Inspection")
        self.geometry("1500x950")
        self.minsize(1150, 700)

        self._stale: set = set()
        self.defects: list[str] = []
        self.labels: dict = {}
        self.counts: dict = {}
        self.cfg: dict = {}
        self.cams = infer.CameraSet()
        self.q: queue.Queue = queue.Queue()
        # Cyclic garbage is collected ONLY on the main thread (see pump). Automatic GC can run on any thread, and
        # when it ran on a background thread it finalised Tk objects (tkinter Font.__del__) there, which blocked
        # that thread for good -- seen freezing the PLC worker, 2026-10-03.
        gc.disable()
        self._gc_at = self._gc_full_at = time.monotonic()

        self.settings = D.load_settings()
        # The ONE owner of the PLC link (plc/service.py). The Machine tab only talks to this object.
        _tr, _st, _tg = plc_link(self.settings)
        self.plc = PLCService(PLCClient(_tr, station=_st, target=_tg), poll_s=0.02, status_period_s=0.15)
        self.plc.operator_controls = bool(self.settings.get("plc_operator_controls", False))
        # Auto-connect only to the simulator. A physical COM port is opened only when the operator presses Connect.
        self.plc_autoconnect = (("--selftest" not in sys.argv and self.settings.get("plc_mode") != "serial")
                                if plc_autoconnect is None else plc_autoconnect)
        self.apply_font_scale(self.settings.get("font_scale", 1.0), save=False)
        # Coded machine alarms (alarms.py), persisted through the project's production store.
        applog.setup()
        applog.log("app", "application started", project=D.PROJECT, theme=theme.MODE)
        self.alarms = AL.AlarmManager(on_change=self._on_alarm)
        self.plc.add_listener(applog.plc_listener)
        self._store = None
        self._store_dir = None
        self.production_dir = None                   # override (self-test: a temporary folder)
        self.ui_mode = "engineer" if self.settings.get("ui_mode") == "engineer" else "operator"

        # ---- status bar: what is this station inspecting, and is everything it depends on alive
        head = ctk.CTkFrame(self, fg_color=RAIL, corner_radius=0, height=52)
        head.pack(fill="x")
        ctk.CTkLabel(head, text="■", font=("Segoe UI", 18), text_color=ACC).pack(side="left", padx=(16, 6))
        ctk.CTkLabel(head, text="VISION INSPECTION", font=theme.H2, text_color=INK).pack(side="left", padx=(0, 14))
        self._pmap: dict[str, str] = {}
        self.project = ctk.CTkOptionMenu(head, values=["-"], width=170,
                                         command=self.switch_project)
        self.project.pack(side="left", padx=(0, 6), pady=10)
        ctk.CTkButton(head, text="+ New", width=64, fg_color="transparent",
                      border_width=1, command=self.new_project).pack(side="left", padx=(0, 4))
        ctk.CTkButton(head, text="Import…", width=84, fg_color="transparent",
                      border_width=1, command=self.import_dataset).pack(side="left")
        self.clock = ctk.CTkLabel(head, text="", font=theme.SMALL, text_color=DIM)
        self.clock.pack(side="right", padx=(10, 16))
        self.mode_btn = ctk.CTkButton(head, text="", width=150, border_width=1, fg_color="transparent",
                                      text_color=INK, command=self.toggle_mode)
        self.mode_btn.pack(side="right", padx=(6, 4))
        self.lamps = {}
        for key in ("MODEL", "CAMERAS", "LINE", "PLC"):          # packed from the right: reads PLC LINE CAMERAS MODEL
            self.lamps[key] = StatusLamp(head, key)
            self.lamps[key].pack(side="right", padx=6)
        self._lamp_at = 0.0

        # ---- footer: dataset summary for the active product
        foot = ctk.CTkFrame(self, fg_color=RAIL, corner_radius=0, height=26)
        foot.pack(side="bottom", fill="x")
        self.status = ctk.CTkLabel(foot, text="", text_color=DIM, font=theme.SMALL)
        self.status.pack(side="left", padx=16, pady=2)

        self.tabs = NavShell(self, self.GROUPS, on_show=self.on_page)
        self.tabs.pack(fill="both", expand=True)
        for name in self.TABS:
            self.tabs.add(name)

        self.tab_production = ProductionTab(self, self.tabs.tab("Production"))
        self.tab_history = hmi.HistoryTab(self, self.tabs.tab("History"))
        self.tab_database = hmi.DatabaseTab(self, self.tabs.tab("Database"))
        self.tab_health = hmi.HealthTab(self, self.tabs.tab("Health"))
        self.tab_models = hmi.ModelsTab(self, self.tabs.tab("Models"))
        self.tab_label = LabelTab(self, self.tabs.tab("Label"))
        self.tab_defects = DefectsTab(self, self.tabs.tab("Defects"))
        self.tab_train = TrainTab(self, self.tabs.tab("Train"))
        self.tab_analysis = AnalysisTab(self, self.tabs.tab("Analysis"))
        self.tab_live = LiveTab(self, self.tabs.tab("Live"))
        self.tab_machine = MachineTab(self, self.tabs.tab("Machine"))
        self.tab_bench = BenchTab(self, self.tabs.tab("Camera"))
        self.tab_data = DataTab(self, self.tabs.tab("Data health"))
        self.tab_annotate = annotation_studio.AnnotationTab(self, self.tabs.tab("Annotate"))
        self.tab_settings = SettingsTab(self, self.tabs.tab("Settings"))

        self.bind_all("<MouseWheel>", self._page_wheel, add="+")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.set_mode(self.ui_mode, save=False)
        self.reload()
        self.after(60, self.pump)

    TABS = ("Production", "History", "Database", "Health", "Label", "Defects", "Train", "Analysis", "Models", "Live",
            "Machine", "Camera", "Data health", "Annotate", "Settings")
    # The navigation rail: the same pages, grouped by what the person is doing. The operator sees
    # PRODUCTION only; ENGINEER mode shows everything (nothing is removed, only hidden).
    GROUPS = (("PRODUCTION", ("Production", "History", "Database", "Health")),
              ("DATA", ("Label", "Defects", "Annotate", "Data health")),
              ("MODEL", ("Train", "Analysis", "Models")),
              ("ENGINEERING", ("Live", "Machine", "Camera")),
              ("SYSTEM", ("Settings",)))
    OPERATOR_PAGES = ("Production", "History", "Database", "Health")

    def _page_wheel(self, event):
        """Wheel over the page background scrolls the page. Anything with its own scrolling (text boxes, lists,
        scrollable panels, canvases) keeps its own wheel: we only act when the nearest such widget IS the page."""
        w = event.widget
        while w is not None:
            if isinstance(w, ScrollHost):
                return w.wheel(event)
            if isinstance(w, (tk.Text, tk.Listbox, ctk.CTkScrollableFrame, ctk.CTkTextbox)):
                return None
            if isinstance(w, tk.Canvas) and not isinstance(w.master, ScrollHost):
                return None                                     # a chart / video canvas: not ours
            w = getattr(w, "master", None)

    def all_tabs(self):
        return (self.tab_production, self.tab_history, self.tab_database, self.tab_health, self.tab_label, self.tab_defects,
                self.tab_train, self.tab_analysis, self.tab_models, self.tab_live, self.tab_machine,
                self.tab_bench, self.tab_data, self.tab_annotate, self.tab_settings)

    # ------------------------------------------------------------- operator / engineer
    def set_mode(self, mode: str, save: bool = True):
        self.ui_mode = "engineer" if mode == "engineer" else "operator"
        eng = self.ui_mode == "engineer"
        self.tabs.show_only(None if eng else self.OPERATOR_PAGES)
        if not eng and self.tabs.get() not in self.OPERATOR_PAGES:
            self.tabs.set("Production")
        self.mode_btn.configure(text="ENGINEER MODE" if eng else "OPERATOR MODE",
                                text_color=WARN if eng else INK, border_color=WARN if eng else LINE)
        self.tab_production.set_mode(self.ui_mode)
        if save:
            self.settings["ui_mode"] = self.ui_mode
            D.save_settings(self.settings)

    def toggle_mode(self):
        if self.ui_mode == "operator":
            pin = str(self.settings.get("engineer_pin") or "")
            if pin:
                got = ctk.CTkInputDialog(title="Engineer mode", text="Engineer PIN:").get_input()
                if got != pin:
                    return
            self.set_mode("engineer")
        else:
            self.set_mode("operator")

    def _on_alarm(self, a):
        """Every alarm change: persisted in the production record and written to logs/alarm.log."""
        applog.alarm_listener(a)
        if self._store is not None:
            self._store.log_alarm(a)

    # ------------------------------------------------------------- production record
    def store(self) -> "production_store.ProductionStore":
        """The active project's production store (SQLite + evidence). Alarms are persisted through it."""
        folder = Path(self.production_dir) if self.production_dir else D.PROJECT_DIR / "production"
        if self._store is None or self._store_dir != folder:
            if self._store is not None:
                self._store.close()
            self._store = production_store.ProductionStore(
                folder, self.settings.get("evidence_policy", "REJECT_AND_FAULT"),
                on_problem=lambda code, cause: self.alarms.raise_(code, cause, source="production store"))
            self._store_dir = folder
        self._store.policy = self.settings.get("evidence_policy", "REJECT_AND_FAULT")
        return self._store

    def apply_font_scale(self, scale: float, save: bool = True):
        """One knob for text size: CustomTkinter scales fonts with the widgets,
        so scaling both keeps padding proportional instead of leaving big text
        clipped inside buttons sized for small text."""
        scale = max(0.7, min(1.8, float(scale)))
        self.applied_scale = scale
        ctk.set_widget_scaling(scale)
        ctk.set_window_scaling(scale)
        if save:
            self.settings["font_scale"] = round(scale, 2)
            D.save_settings(self.settings)

    # ------------------------------------------------------------- plumbing
    def pump(self):
        """Single drain point for every background thread."""
        try:
            while True:
                fn = self.q.get_nowait()
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        now = time.monotonic()
        if now - self._lamp_at >= 0.5:
            self._lamp_at = now
            try:
                self.update_lamps()
            except Exception:
                traceback.print_exc()
        if now - self._gc_at >= 1.0:                     # the only place cyclic GC runs
            self._gc_at = now
            if now - self._gc_full_at >= 30.0:
                self._gc_full_at = now
                gc.collect()
            else:
                gc.collect(1)
        self.after(60, self.pump)

    def update_lamps(self):
        """Status-bar lamps. Cached state only (PLCService.link_state, Camera.alive): never any device I/O."""
        st = self.plc.link_state()
        self.lamps["PLC"].show({PLC_CONNECTED: GOOD, PLC_DEGRADED: WARN, PLC_FAULT: BAD}.get(st, OFF),
                               {PLC_CONNECTED: "OK", PLC_DEGRADED: "SLOW", PLC_FAULT: "FAULT"}.get(st, "OFF"))
        line = getattr(self, "tab_production", None)
        st = line.mstate if line is not None else MS.OFFLINE          # the ONE machine state (machine_state.py)
        cls = MS.LOOK.get(st, ("OFF", ""))[0]
        self.lamps["LINE"].show({"PASS": GOOD, "BAD": BAD, "WARN": WARN, "ACC": ACC}.get(cls, OFF), st.replace("_", " "))
        n = len(self.cams.running())
        self.lamps["CAMERAS"].show(GOOD if n else OFF, f"{n} ON" if n else "OFF")
        model = self.cfg.get("active_model")
        self.lamps["MODEL"].show(GOOD if model else WARN,
                                 f"{model[2:8]} {model[9:13]}" if model and len(model) >= 13 else (model or "NONE"))
        self.clock.configure(text=time.strftime("%d %b  %H:%M"))

    def post(self, fn):
        self.q.put(fn)

    def run_bg(self, work, done=None):
        def target():
            try:
                r = work()
            except Exception as e:
                self.post(lambda e=e: messagebox.showerror("Error", f"{type(e).__name__}: {e}"))
                return
            if done:
                self.post(lambda: done(r))
        threading.Thread(target=target, daemon=True).start()

    # -------------------------------------------------------------- projects
    def switch_project(self, title):
        name = self._pmap.get(title)
        if not name or name == D.PROJECT:
            return
        # The camera thread is holding the old project's model and thresholds;
        # let it keep running and the Live tab scores frames against a model
        # that no longer belongs to the data on screen. The inspection line goes first: it is
        # driving those cameras and logging into the old project's folder.
        self.tab_production.stop_line()
        self.cams.stop()
        D.use_project(name)
        self.reload()

    # Task labels shown in the dialog -> the value dataset.create_project()
    # accepts. A fixed dropdown, not free text, is what keeps an invalid task
    # from ever reaching create_project() -- its own validation is the
    # backstop, this just means it's never exercised by a typo here.
    TASK_LABELS = {"Classification": "classification", "Detection": "detection",
                   "Segmentation": "segmentation"}

    def new_project(self):
        result: dict = {}
        win = ctk.CTkToplevel(self)
        win.title("New project")
        win.geometry("360x260")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(win, text="Name for the new project", font=("Segoe UI", 15)).pack(
            padx=20, pady=(20, 6), anchor="w")
        name_entry = ctk.CTkEntry(win, width=320)
        name_entry.pack(padx=20)
        name_entry.focus()

        ctk.CTkLabel(win, text="Task", font=("Segoe UI", 15)).pack(
            padx=20, pady=(16, 6), anchor="w")
        task_menu = ctk.CTkOptionMenu(win, width=320, values=list(self.TASK_LABELS))
        task_menu.set("Classification")
        task_menu.pack(padx=20)

        def submit():
            result["title"] = name_entry.get()
            result["task"] = task_menu.get()
            win.destroy()

        name_entry.bind("<Return>", lambda e: submit())
        btns = ctk.CTkFrame(win, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(20, 16))
        ctk.CTkButton(btns, text="Cancel", width=90, fg_color="transparent", border_width=1,
                      command=win.destroy).pack(side="left")
        ctk.CTkButton(btns, text="Create", width=90, fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=submit).pack(side="right")
        win.after(150, win.lift)
        win.wait_window()

        title = result.get("title")
        if not title or not title.strip():
            return
        task = self.TASK_LABELS.get(result.get("task"), D.DEFAULT_TASK)

        try:
            name = D.create_project(title, task=task)
        except ValueError as e:
            return messagebox.showerror("Cannot create", str(e))
        self.open_project(name)
        if task == "classification":
            hint = (f"{title!r} is empty.\n\nGo to Defect types: 'Upload GOOD images' for bottles "
                    f"that pass, then add a defect type and use its Upload button for the ones "
                    f"that fail.")
        else:
            hint = (f"{title!r} is empty and set up for {task}.\n\nAdd images under its images/ "
                    f"folder, then use the Annotate tab to label them.")
        messagebox.showinfo("Project created", hint)

    def import_dataset(self):
        ImportDialog(self)

    def reload(self):
        names = D.list_projects()
        self._pmap = {D.project_title(n): n for n in names}
        self.project.configure(values=list(self._pmap) or ["-"])
        self.project.set(D.project_title(D.PROJECT))

        self.defects, self.labels = D.load_labels()
        self.cfg = D.load_config()
        self.refresh_summary()
        # Only the page on screen is rebuilt. The other ten were costing ~20 s at start-up (3,000+ widgets, most of
        # them on pages nobody is looking at); each is refreshed the first time it is shown instead.
        self._stale = set(self.TABS)
        self.on_page(self.tabs.get() or self.TABS[0])

    def refresh_summary(self):
        self.counts = c = D.counts(self.defects, self.labels)
        model = self.cfg.get("active_model") or "no model"
        self.status.configure(
            text=f"{D.project_title(D.PROJECT)}   ·   {c['_total']} images   ·   {c['_good']} good   ·   "
                 f"{c['_defective']} defective   ·   {c['_unreviewed']} unreviewed   ·   model {model}")

    # Tabs whose view is built from labels.csv. A label edit marks them stale instead of rebuilding them;
    # each is refreshed the next time it is shown.
    LABEL_VIEWS = ("Defects", "Train", "Data health")

    def data_changed(self):
        """labels.csv changed under the Label tab: re-read it, update the summary, mark dependent tabs stale."""
        self.defects, self.labels = D.load_labels()
        self.refresh_summary()
        self._stale.update(self.LABEL_VIEWS)

    def tab_of(self, name: str):
        return dict(zip(self.TABS, self.all_tabs()))[name]

    def on_page(self, name: str):
        if name in self._stale:
            self._stale.discard(name)
            self.tab_of(name).refresh()

    def open_project(self, name: str):
        """Switch to a project. The line and the cameras stop first: they hold the old project's model and log."""
        self.tab_production.stop_line()
        self.cams.stop()
        D.use_project(name)
        self.reload()

    def restart(self):
        """Relaunch the app (used to apply a new text size). The line is stopped first, so it asks."""
        if self.tab_production.running and not messagebox.askyesno(
                "Restart", "The inspection line is running. Restart anyway? It will be stopped."):
            return
        self.on_close()
        import os
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def on_close(self):
        self.tab_production.close()          # stops the machine cycle before the cameras and the PLC link
        self.cams.stop()
        self.tab_machine.close()
        self.plc.stop()                      # sends nothing; the PLC clears M0/M1 itself
        if self._store is not None:
            self._store.close()
        applog.log("app", "application closed")
        self.destroy()


class ImportDialog:
    """Import a dataset that is already sorted one sub-folder per class.

    This is what makes the station product-agnostic: a new object is a new
    product (project) plus its folders. Every class folder becomes a defect
    column, a good/ok folder becomes GOOD, anything unsure goes to the inbox.
    Files are copied, never moved (D.run_import -> D.add_images).
    """
    CHOICES = ("GOOD", "INBOX", "SKIP")

    def __init__(self, app, src=None, modal=True):
        self.app, self.modal, self.result = app, modal, None
        src = src or filedialog.askdirectory(title="Folder with one sub-folder per class")
        if not src:
            self.win = None
            return
        self.src = Path(src)
        try:
            self.plan = D.plan_import(self.src)
        except ValueError as e:
            self.win = None
            messagebox.showerror("Import", str(e))
            return
        if not self.plan:
            self.win = None
            messagebox.showinfo("Import", f"No .jpg / .png images found in\n{self.src}")
            return

        self.win = win = ctk.CTkToplevel(app)
        win.title("Import dataset")
        win.geometry("760x620")
        win.transient(app)
        ctk.CTkLabel(win, text="IMPORT DATASET", font=theme.CAPS, text_color=DIM).pack(anchor="w", padx=18,
                                                                                      pady=(16, 0))
        ctk.CTkLabel(win, text=str(self.src), font=theme.SMALL, text_color=INK, wraplength=700,
                     justify="left").pack(anchor="w", padx=18)

        tgt = ctk.CTkFrame(win, fg_color=PANEL)
        tgt.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(tgt, text="Into", font=theme.H2).pack(side="left", padx=(12, 10), pady=10)
        self.where = ctk.StringVar(value="new")
        ctk.CTkRadioButton(tgt, text="a NEW product:", variable=self.where, value="new",
                           command=self._summary).pack(side="left")
        self.name = ctk.CTkEntry(tgt, width=200)
        self.name.insert(0, self.src.name)
        self.name.pack(side="left", padx=(4, 16))
        ctk.CTkRadioButton(tgt, text=f"current: {D.project_title(D.PROJECT)}", variable=self.where, value="cur",
                           command=self._summary).pack(side="left")

        ctk.CTkLabel(win, text="Each folder becomes…   (type a new name to create a class)", font=theme.SMALL,
                     text_color=DIM).pack(anchor="w", padx=18, pady=(4, 2))
        rows = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        rows.pack(fill="both", expand=True, padx=16)
        values = list(self.CHOICES) + sorted(set(app.defects) | {r["guess"] for r in self.plan} - set(self.CHOICES))
        self.combo = {}
        for r in self.plan:
            line = ctk.CTkFrame(rows, fg_color="transparent")
            line.pack(fill="x", pady=2)
            ctk.CTkLabel(line, text=("(loose files)" if r["folder"] == "." else r["folder"]), width=260,
                         anchor="w").pack(side="left", padx=(8, 4))
            ctk.CTkLabel(line, text=f"{r['n']} img", width=70, anchor="e", text_color=DIM).pack(side="left")
            cb = ctk.CTkComboBox(line, values=values, width=240, command=lambda _: self._summary())
            cb.set(r["guess"])
            cb.bind("<KeyRelease>", lambda e: self._summary())
            cb.pack(side="left", padx=12)
            self.combo[r["folder"]] = cb

        self.sum = ctk.CTkLabel(win, text="", font=theme.SMALL, text_color=DIM, justify="left", wraplength=700)
        self.sum.pack(anchor="w", padx=18, pady=(8, 4))
        btns = ctk.CTkFrame(win, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(4, 16))
        ctk.CTkButton(btns, text="Cancel", width=90, fg_color="transparent", border_width=1,
                      command=self.close).pack(side="left")
        self.go = ctk.CTkButton(btns, text="Import", width=110, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                                command=self.run)
        self.go.pack(side="right")
        self._summary()
        if modal:
            win.after(150, win.lift)
            win.grab_set()

    def mapping(self) -> dict:
        out = {}
        for folder, cb in self.combo.items():
            v = cb.get().strip()
            out[folder] = v.upper() if v.upper() in self.CHOICES else D.slug(v)
        return out

    def _summary(self):
        m = self.mapping()
        n = {r["folder"]: r["n"] for r in self.plan}
        good = sum(n[f] for f, t in m.items() if t == "GOOD")
        inbox = sum(n[f] for f, t in m.items() if t == "INBOX")
        classes = sorted({t for t in m.values() if t not in self.CHOICES and t})
        bad = [f for f, t in m.items() if not t or t in D.RESERVED]
        new = [c for c in classes if self.where.get() == "new" or c not in self.app.defects]
        txt = (f"{good} GOOD   ·   {inbox} to inbox   ·   {len(classes)} defect classes "
               f"({sum(n[f] for f, t in m.items() if t in classes)} images)")
        if new:
            txt += f"\nnew classes: {', '.join(new)}"
        if good < 50:
            txt += "\n⚠ fewer than 50 GOOD images: the model will barely see a passing part."
        if bad:
            txt += f"\n✖ give these folders a target: {', '.join(bad)}"
        self.sum.configure(text=txt, text_color=BAD if bad else DIM)
        self.go.configure(state="disabled" if bad else "normal")

    def close(self):
        if self.win is not None:
            self.win.destroy()
            self.win = None

    def run(self):
        m = self.mapping()
        new = self.where.get() == "new"
        if new:
            try:
                name = D.create_project(self.name.get())
            except ValueError as e:
                return messagebox.showerror("Cannot create", str(e), parent=self.win)
            self.app.open_project(name)
        self.go.configure(state="disabled", text="Copying…")
        src = self.src

        def done(got):
            self.close()
            self.app.reload()
            self.result = got
            if not self.modal:                   # selftest / scripted use: no blocking dialogs
                return
            lines = "\n".join(f"  {t}: {n}" for t, n in sorted(got.items()))
            msg = f"Copied into {D.project_title(D.PROJECT)}:\n{lines}"
            if new and messagebox.askyesno(
                    "Imported", msg + "\n\nA new product needs its own crop region (ROI). Measure it now from "
                                      "these images?\n\n(Data health → Re-measure does the same later.)"):
                self.app.tabs.set("Data health")
                self.app.tab_data.recalibrate()
            elif not new:
                messagebox.showinfo("Imported", msg)
        self.app.run_bg(lambda: D.run_import(src, m), done)


# ------------------------------------------------------------------- Machine
class ConveyorHMI(ctk.CTkToplevel):
    """Small animated view of the line. Lamps, belt motion, sensor beam and reject pusher follow the real
    PLC bits from PLCService's cached snapshot. The bottle's position is an illustration derived from those
    bits (the PLC does not report where a bottle is)."""

    W, H = 660, 330
    BELT_Y, X_IN, X_STATION, X_PUSH, X_OUT = 170, 40, 220, 450, 620
    SPEED = 130.0                                    # px/s, illustration only

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Conveyor HMI")
        self.geometry(f"{self.W + 20}x{self.H + 20}")
        self.resizable(False, False)
        self.cv = ctk.CTkCanvas(self, width=self.W, height=self.H, bg=BG, highlightthickness=0)
        self.cv.pack(padx=10, pady=10)
        self.bottle = None                           # {"x", "y", "mode"}: station / pass / reject / pushed
        self.phase = 0.0
        self.note = ""
        self._t = time.monotonic()
        self._alive = True
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self._tick)

    def close(self):
        self._alive = False
        self.destroy()

    def _tick(self):
        if not self._alive:
            return
        try:
            self.step()
        except Exception:
            traceback.print_exc()
        self.after(50, self._tick)

    def step(self):
        now = time.monotonic()
        dt, self._t = min(0.2, now - self._t), now
        svc = self.app.plc
        snap = svc.snapshot() if svc.link_state() == PLC_CONNECTED else None
        f = {}
        if snap:
            for k in ("inputs", "internal", "outputs", "timers", "counters"):
                f.update(snap[k])
            self._bottle(f, dt)
            if f["Y1"]:
                self.phase = (self.phase + self.SPEED * dt) % 40
        else:
            self.bottle = None
        self.draw(f, svc.link_state())

    def _bottle(self, f, dt):
        b = self.bottle
        rejecting = bool(f["M1"] or f["T0"] or f["Y0"] or f["T1"])
        if f["M2"] and (b is None or b["mode"] == "pass"):
            b = self.bottle = {"x": float(self.X_STATION), "y": 0.0, "mode": "station"}
            self.note = ""
        if b is None:
            return
        if b["mode"] == "station" and not f["M2"]:
            b["mode"] = "reject" if rejecting else "pass"
        if b["mode"] == "pass":
            if rejecting:
                b["mode"] = "reject"
            elif f["Y1"]:
                b["x"] += self.SPEED * dt
                if b["x"] > self.X_OUT + 20:
                    self.bottle = None
        elif b["mode"] == "reject":
            if f["Y0"]:
                b["mode"], b["x"] = "pushed", float(self.X_PUSH)
            elif not rejecting:                      # sequence ended and Y0 was never seen ON
                b["mode"], self.note = "pass", "reject sequence ended - Y0 was not observed ON"
            else:
                b["x"] = min(float(self.X_PUSH), b["x"] + self.SPEED * dt)
        elif b["mode"] == "pushed":
            b["y"] += 220 * dt
            if b["y"] > 95 and not f["Y0"]:
                self.bottle = None

    def draw(self, f, link):
        c, y = self.cv, self.BELT_Y
        c.delete("all")
        on = lambda k: bool(f.get(k))
        c.create_text(12, 14, anchor="w", text="CONVEYOR", font=("Segoe UI", 14, "bold"), fill=INK)
        c.create_text(self.W - 12, 14, anchor="e", font=("Segoe UI", 13, "bold"),
                      text=("RUNNING" if on("Y1") else "STOPPED") if f else f"NO PLC LINK ({link})",
                      fill=(GOOD if on("Y1") else DIM) if f else BAD)
        # belt + moving stripes + rollers
        c.create_rectangle(self.X_IN, y, self.X_OUT, y + 26, fill=PANEL_2, outline=DIM, width=2)
        x = self.X_IN + self.phase
        while x < self.X_OUT - 4:
            c.create_line(x, y + 4, x + 10, y + 22, fill=DIM, width=2)
            x += 40
        for rx in (self.X_IN, self.X_OUT):
            c.create_oval(rx - 13, y, rx + 13, y + 26, fill=OFF, outline=LINE, width=2)
        # sensor (X0) and camera (M2) at the inspection station
        sx = self.X_STATION
        c.create_rectangle(sx - 34, y - 76, sx - 26, y, fill=GOOD if on("X0") else OFF, outline="")
        if on("X0"):
            c.create_line(sx - 26, y - 30, sx + 30, y - 30, fill=BAD, width=2, dash=(4, 3))
        c.create_text(sx - 30, y - 86, text="X0 sensor", font=("Segoe UI", 12), fill=GOOD if on("X0") else DIM)
        c.create_rectangle(sx - 18, 36, sx + 18, 60, fill=WARN if on("M2") else OFF, outline="")
        c.create_polygon(sx - 8, 60, sx + 8, 60, sx + 14, 72, sx - 14, 72, fill=WARN if on("M2") else OFF)
        c.create_text(sx + 26, 48, anchor="w", font=("Segoe UI", 12, "bold"),
                      text="INSPECT (M2)" if on("M2") else "camera", fill=WARN if on("M2") else DIM)
        # reject pusher (Y0) and bin
        px, ext = self.X_PUSH, 34 if on("Y0") else 0
        c.create_rectangle(px - 16, y - 96, px + 16, y - 70, fill=LINE, outline="")
        c.create_rectangle(px - 4, y - 70, px + 4, y - 58 + ext, fill=BAD if on("Y0") else OFF, outline="")
        c.create_rectangle(px - 13, y - 58 + ext, px + 13, y - 50 + ext, fill=BAD if on("Y0") else OFF, outline="")
        c.create_text(px + 22, y - 84, anchor="w", text="Y0 reject", font=("Segoe UI", 12, "bold" if on("Y0") else "normal"),
                      fill=BAD if on("Y0") else DIM)
        c.create_rectangle(px - 30, y + 44, px + 30, y + 110, outline=DIM, width=2)
        c.create_text(px, y + 120, text="reject bin", font=("Segoe UI", 12), fill=DIM)
        c.create_text(self.X_OUT - 6, y + 44, anchor="e", text="pass ->", font=("Segoe UI", 12), fill=DIM)
        if f:                                            # PLC counters: C0 = PASS commands, C1 = REJECT commands
            c.create_text(px, y + 77, text=str(f["C1"]), font=("Segoe UI", 18, "bold"), fill=BAD)
            c.create_text(self.X_OUT - 6, y + 64, anchor="e", text=f"PASS  {f['C0']}", font=("Segoe UI", 15, "bold"), fill=GOOD)
            c.create_text(self.X_OUT - 6, y + 84, anchor="e", text=f"total {f['C0'] + f['C1']}", font=("Segoe UI", 12), fill=DIM)
        # bottle
        b = self.bottle
        if b:
            bx, by = b["x"], y - 2 + b["y"]
            col = {"station": WARN, "reject": BAD, "pushed": BAD}.get(b["mode"], ACC)
            c.create_rectangle(bx - 9, by - 40, bx + 9, by, fill=PANEL_2, outline=col, width=2)
            c.create_rectangle(bx - 4, by - 52, bx + 4, by - 40, fill=PANEL_2, outline=col, width=2)
            c.create_rectangle(bx - 5, by - 57, bx + 5, by - 52, fill=col, outline="")
        # lamps: the real bits
        lamps = (("X0", "sensor"), ("M2", "trigger"), ("M0", "pass"), ("M1", "reject"), ("Y1", "conveyor"), ("Y0", "solenoid"))
        for i, (k, name) in enumerate(lamps):
            lx = 22 + i * 106
            col = (BAD if k in ("M1", "Y0") else GOOD) if on(k) else PANEL_2
            c.create_oval(lx, self.H - 30, lx + 14, self.H - 16, fill=col, outline=DIM)
            c.create_text(lx + 20, self.H - 23, anchor="w", text=f"{k} {name}", font=("Segoe UI", 12), fill=INK if on(k) else DIM)
        if f:
            c.create_text(12, self.H - 46, anchor="w", font=("Consolas", 12), fill=DIM,
                          text=f"T0 {f['T0']}   T1 {f['T1']}   {self.note}")
        c.create_text(self.W - 12, 32, anchor="e", font=("Segoe UI", 11), fill=DIM,
                      text="lamps = real PLC bits; bottle position is illustrative")


class MachineTab:
    """Machine / PLC status. Reads the app's single PLCService; it never touches PLCClient.

    Python owns: receiving the M2 trigger, the PASS/REJECT decision, sending M0/M1, showing state.
    The PLC ladder owns: conveyor Y1, reject delay T0, solenoid Y0, pulse T1. Nothing here writes Y or T.
    The simulator switches and the test command are commissioning tools, marked as such.
    """

    NAMES = {"X0": "Bottle Sensor", "X1": "Start", "X2": "Stop", "M2": "Inspection Trigger",
             "M0": "PASS Command", "M1": "REJECT Command", "Y1": "Conveyor", "Y0": "Reject Solenoid",
             "T0": "Reject Travel Delay", "T1": "Reject Pulse", "C0": "PASS count", "C1": "REJECT count"}
    GROUPS = (("INPUTS", ("X0", "X1", "X2")), ("INTERNAL", ("M2", "M0", "M1")),
              ("OUTPUTS", ("Y1", "Y0")), ("TIMERS (x0.1 s)", ("T0", "T1")), ("COUNTERS", ("C0", "C1")))
    SIM = ("X0", "X1", "X2", "M2", "M0", "M1")       # simulator-only switches
    MODES = {"Simulator (TCP)": "tcp", "Real PLC (serial)": "serial"}
    LOG_LABEL = {"X0": "X0 SENSOR", "X1": "X1 START", "X2": "X2 STOP", "M2": "M2 TRIGGER", "M0": "M0 PASS CMD",
                 "M1": "M1 REJECT CMD", "Y0": "Y0 REJECT", "Y1": "Y1 CONVEYOR", "T0": "T0 DELAY", "T1": "T1 PULSE",
                 "C0": "C0 PASS COUNT", "C1": "C1 REJ COUNT"}

    def __init__(self, app: App, parent):
        self.app = app
        self.cells: dict = {}
        self.last_cmd = None
        self._last_res = None
        self.cmd_busy = False
        self.msg = ("", True)
        self._lock = threading.Lock()
        self._log_n = -1
        self._closed = False
        self._started = False
        self._hold: dict = {}                        # device -> monotonic time until which the switch is not mirrored
        self.hmi = None

        head = ctk.CTkFrame(parent, fg_color=PANEL)
        head.pack(fill="x", pady=(0, 6))
        self.state = ctk.CTkLabel(head, text="DISCONNECTED", font=("Segoe UI", 18, "bold"), text_color=DIM, width=170)
        self.state.pack(side="left", padx=(12, 8), pady=8)
        self.run_lbl = ctk.CTkLabel(head, text="PLC --", font=("Segoe UI", 15, "bold"), text_color=DIM, width=110)
        self.run_lbl.pack(side="left")
        # which machine this is talking to, impossible to miss: a simulator test must never be mistaken for the real PLC
        self.where_lbl = ctk.CTkLabel(head, text="", font=theme.CAPS, text_color=ACC_T, fg_color=OFF, corner_radius=6,
                                      width=110)
        self.where_lbl.pack(side="left", padx=(0, 8))
        self.info = ctk.CTkLabel(head, text="", text_color=DIM, font=MONO, justify="left", anchor="w")
        self.info.pack(side="left", padx=10)
        ctk.CTkButton(head, text="Disconnect", width=96, fg_color="transparent", border_width=1, text_color=INK,
                      command=self.disconnect).pack(side="right", padx=(4, 12))
        ctk.CTkButton(head, text="Connect", width=90, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.connect).pack(side="right", padx=4)
        ctk.CTkButton(head, text="Conveyor HMI", width=120, command=self.open_hmi).pack(side="right", padx=4)
        if app.settings.get("show_simulation_check", True):
            ctk.CTkButton(head, text="Simulation check", width=130, command=self.open_simcheck).pack(side="right", padx=4)

        cfg = app.settings
        conn = ctk.CTkFrame(parent, fg_color=PANEL)
        conn.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(conn, text="PLC LINK", font=("Segoe UI", 13, "bold"), text_color=DIM).pack(side="left", padx=(12, 8), pady=8)
        self.mode = ctk.CTkOptionMenu(conn, values=list(self.MODES), width=160, command=lambda _=None: self._mode_changed())
        self.mode.set(next(k for k, v in self.MODES.items() if v == ("serial" if cfg.get("plc_mode") == "serial" else "tcp")))
        self.mode.pack(side="left")
        holder = ctk.CTkFrame(conn, fg_color="transparent")
        holder.pack(side="left", padx=6)
        self.f_tcp = ctk.CTkFrame(holder, fg_color="transparent")
        ctk.CTkLabel(self.f_tcp, text="Host").pack(side="left", padx=(4, 4))
        self.host = ctk.CTkEntry(self.f_tcp, width=110)
        self.host.insert(0, str(cfg.get("plc_host", "127.0.0.1")))
        self.host.pack(side="left")
        ctk.CTkLabel(self.f_tcp, text="Port").pack(side="left", padx=(8, 4))
        self.port = ctk.CTkEntry(self.f_tcp, width=64)
        self.port.insert(0, str(cfg.get("plc_port", 10002)))
        self.port.pack(side="left")
        self.f_ser = ctk.CTkFrame(holder, fg_color="transparent")
        self.com = ctk.CTkOptionMenu(self.f_ser, values=["-"], width=230)
        self.com.pack(side="left", padx=(4, 2))
        ctk.CTkButton(self.f_ser, text="Scan", width=50, command=self.scan_ports).pack(side="left", padx=2)
        self.baud = ctk.CTkOptionMenu(self.f_ser, values=["9600", "19200", "38400", "57600", "115200"], width=86)
        self.baud.set(str(cfg.get("plc_baud", 9600)))
        self.baud.pack(side="left", padx=2)
        self.fmt = ctk.CTkOptionMenu(self.f_ser, values=sorted(SERIAL_FORMATS), width=70)
        self.fmt.set(str(cfg.get("plc_format", "7E1")))
        self.fmt.pack(side="left", padx=2)
        ctk.CTkLabel(conn, text="Station").pack(side="left", padx=(4, 4))
        self.station = ctk.CTkEntry(conn, width=40)
        self.station.insert(0, str(cfg.get("plc_station", 1)))
        self.station.pack(side="left")
        ctk.CTkButton(conn, text="Test link", width=80, fg_color="transparent", border_width=1, text_color=INK,
                      command=self.test_link).pack(side="left", padx=8)
        self.conn_note = ctk.CTkLabel(conn, text="", text_color=DIM, font=("Segoe UI", 13), anchor="w", justify="left")
        self.conn_note.pack(side="left", padx=4)
        self.auto = ctk.CTkCheckBox(conn, text="auto-reconnect", width=120)
        if cfg.get("plc_auto_reconnect", True):
            self.auto.select()
        self.auto.pack(side="right", padx=10)
        self._com_map: dict = {}
        self.scan_ports(select=str(cfg.get("plc_com", "")))
        self._mode_changed()
        # proof-of-life line: reply counter and data age tick while the link is real; endpoint = what answers
        self.live_lbl = ctk.CTkLabel(parent, text="", font=MONO, text_color=DIM, anchor="w", justify="left")
        self.live_lbl.pack(fill="x", padx=6, pady=(0, 4))
        self._want_link = False                      # the operator asked for a link (Connect) and has not pressed Disconnect
        self._last_auto = 0.0
        self._op_busy = False
        self._ep = (0.0, None)

        self.flow_lbl = ctk.CTkLabel(parent, text="", font=("Segoe UI", 15, "bold"), text_color=DIM, anchor="w")
        self.flow_lbl.pack(fill="x", padx=6, pady=(0, 6))

        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 6))
        for col, (title, names) in enumerate(self.GROUPS):
            box = ctk.CTkFrame(grid, fg_color=PANEL)
            box.grid(row=0, column=col, sticky="nsew", padx=(0, 6))
            grid.grid_columnconfigure(col, weight=1, uniform="g")
            ctk.CTkLabel(box, text=title, font=("Segoe UI", 13, "bold"), text_color=DIM).pack(anchor="w", padx=10, pady=(8, 2))
            for n in names:
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=(1, 3))
                ctk.CTkLabel(row, text=n, width=34, anchor="w", font=("Consolas", 13, "bold")).pack(side="left")
                v = ctk.CTkLabel(row, text="--", width=50, corner_radius=6, fg_color=OFF, text_color=ACC_T)
                v.pack(side="left", padx=4)
                ctk.CTkLabel(row, text=self.NAMES[n], text_color=DIM, font=("Segoe UI", 13)).pack(side="left")
                self.cells[n] = v

        ctl = ctk.CTkFrame(parent, fg_color=PANEL)
        ctl.pack(fill="x", pady=(0, 6))
        row = ctk.CTkFrame(ctl, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(row, text="SIMULATOR TEST", font=("Segoe UI", 13, "bold"), text_color=WARN, width=120,
                     anchor="w").pack(side="left")
        self.sw = {}
        for dev in self.SIM:
            var = ctk.BooleanVar(value=False)
            sw = ctk.CTkSwitch(row, text=dev, variable=var, width=70,
                               command=lambda d=dev, v=var: self.sim_input(d, v.get()))
            sw.pack(side="left", padx=(0, 6))
            self.sw[dev] = (sw, var)
        self.pulse_btn = ctk.CTkButton(row, text="Pulse X0", width=84, command=self.pulse_x0)
        self.pulse_btn.pack(side="left", padx=8)
        row = ctk.CTkFrame(ctl, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(row, text="TEST COMMAND", font=("Segoe UI", 13, "bold"), text_color=WARN, width=120,
                     anchor="w").pack(side="left")
        self.arm = ctk.CTkCheckBox(row, text="arm", width=60)
        self.arm.pack(side="left", padx=(0, 6))
        self.btn_pass = ctk.CTkButton(row, text="PASS (M0)", width=100, fg_color=GOOD, text_color=ACC_T, state="disabled",
                                      command=lambda: self.send("PASS"))
        self.btn_pass.pack(side="left", padx=4)
        self.btn_rej = ctk.CTkButton(row, text="REJECT (M1)", width=110, fg_color=BAD, text_color=ACC_T, state="disabled",
                                     command=lambda: self.send("REJECT"))
        self.btn_rej.pack(side="left", padx=4)
        self.cmd_lbl = ctk.CTkLabel(row, text="", font=MONO, anchor="w", justify="left")
        self.cmd_lbl.pack(side="left", padx=10)
        # real-PLC operator test controls (commissioning only; settings.json "plc_operator_controls")
        row = ctk.CTkFrame(ctl, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(row, text="OPERATOR TEST", font=("Segoe UI", 13, "bold"), text_color=WARN, width=120,
                     anchor="w").pack(side="left")
        self.op_arm = ctk.CTkCheckBox(row, text="arm", width=60)
        self.op_arm.pack(side="left", padx=(0, 6))
        self.op_btns = {}
        for action, label in (("START", "Conveyor START (M10)"), ("STOP", "Conveyor STOP (M11)"),
                              ("TRIGGER", "Virtual bottle (M2)")):
            b = ctk.CTkButton(row, text=label, width=150, state="disabled",
                              command=lambda a=action: self.operator(a))
            b.pack(side="left", padx=4)
            self.op_btns[action] = b
        self.op_note = ctk.CTkLabel(row, text="", text_color=DIM, font=("Segoe UI", 12), anchor="w")
        self.op_note.pack(side="left", padx=10)

        self.log = ctk.CTkTextbox(parent, font=MONO, fg_color=PANEL, text_color=INK, height=170)
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")
        app.after(250, self._tick)
        if app.plc_autoconnect:
            app.after(400, self.connect)

    # ------------------------------------------------------------ actions (never on the UI thread)
    def _op(self, name, fn):
        def run():
            try:
                out = fn()
                msg = (out if isinstance(out, str) else "", True)
            except Exception as e:                       # noqa: BLE001 - shown, never swallowed
                msg = (f"{name} FAILED: {type(e).__name__}: {e}", False)
            with self._lock:
                self.msg = msg
                self._op_busy = False
        self._op_busy = True
        threading.Thread(target=run, daemon=True).start()

    def disconnect(self):
        self._want_link = False                      # stops auto-reconnect until Connect is pressed again
        self._op("disconnect", self.app.plc.disconnect)

    # ------------------------------------------------------------ link configuration
    def _mode_changed(self):
        serial = self.MODES[self.mode.get()] == "serial"
        (self.f_tcp if serial else self.f_ser).pack_forget()
        (self.f_ser if serial else self.f_tcp).pack(side="left")
        self.conn_note.configure(
            text=("Physical PLC: serial link NOT yet verified on hardware. One program per COM port - "
                  "go offline in ISPSoft/COMMGR first.") if serial else "ISPSoft simulator (COMMGR 'Simulation SE').",
            text_color=WARN if serial else DIM)

    def scan_ports(self, select: str = ""):
        ports = serial_ports()
        names = [f"{dev} - {desc}"[:44] for dev, desc in ports] or ["- no COM port found -"]
        self._com_map = {n: dev for n, (dev, _) in zip(names, ports)}
        self.com.configure(values=names)
        self.com.set(next((n for n, d in self._com_map.items() if d == select), names[0]))

    def link_from_ui(self) -> dict:
        """The link settings as typed. Raises ValueError with a readable message; opens nothing."""
        try:
            station = int(self.station.get())
            if not 1 <= station <= 247:
                raise ValueError
        except ValueError:
            raise ValueError("station must be a number 1..247") from None
        if self.MODES[self.mode.get()] == "serial":
            dev = self._com_map.get(self.com.get())
            if not dev:
                raise ValueError("no COM port selected (plug in the PLC cable, then Scan)")
            return {"plc_mode": "serial", "plc_com": dev, "plc_baud": int(self.baud.get()),
                    "plc_format": self.fmt.get(), "plc_station": station}
        try:
            port = int(self.port.get())
        except ValueError:
            raise ValueError("TCP port must be a number") from None
        if not self.host.get().strip():
            raise ValueError("host is empty")
        return {"plc_mode": "tcp", "plc_host": self.host.get().strip(), "plc_port": port, "plc_station": station}

    def connect(self, apply_ui: bool = True):
        """Initialise the link chosen above and prove the PLC answers. apply_ui=False keeps the service's
        current transport (used by the self-test, which runs against a fake PLC)."""
        svc = self.app.plc
        link = None
        if apply_ui:
            try:
                cfg = self.link_from_ui()
            except ValueError as e:
                with self._lock:
                    self.msg = (f"connect: {e}", False)
                return
            self.app.settings.update(cfg)
            D.save_settings(self.app.settings)
            link = plc_link(self.app.settings)
        self._want_link = True
        if not self._started or svc is not getattr(self, "_started_svc", None):
            svc.start()
            self._started, self._started_svc = True, svc

        def run():
            if link is not None:
                svc.reconfigure(*link)
            lat = svc.connect()
            return f"connected to {svc.client.transport.description}: PLC answered in {lat:.0f} ms"
        self._op("connect", run)

    def test_link(self):
        def run():
            r = self.app.plc.link_test(10)
            return (f"link test: {r['replies']}/10 replies, median {r['median_ms']:.0f} ms, max {r['max_ms']:.0f} ms, "
                    f"PLC {'RUN' if r['plc_run'] else 'NOT RUN'}")
        self._op("link test", run)

    def open_simcheck(self):
        if self.app.ui_mode != "engineer":
            return messagebox.showinfo("Engineer only", "The simulation check is an engineering tool.")
        import hmi as hmi_mod                                          # "hmi" is also the conveyor window name below
        hmi_mod.SimulationCheckDialog(self.app)

    def open_hmi(self):
        if self.hmi is None or not self.hmi._alive:
            self.hmi = ConveyorHMI(self.app)
        self.hmi.lift()

    def sim_input(self, dev, on):
        """Simulator-only: X0/X1/X2 stand in for the sensor and buttons, M2/M0/M1 for the handshake bits."""
        if not self.app.plc.simulator_mode:
            return
        self._hold[dev] = time.monotonic() + 0.7
        self._op(f"simulator {dev} <- {int(on)}", lambda: self.app.plc.simulator_test_write(dev, bool(on)))

    def operator(self, action):
        """Real-PLC operator test (START/STOP conveyor via M10/M11, virtual bottle via M2). One shot per press."""
        if not self.op_arm.get():
            return
        self.op_arm.deselect()                       # one press -> one write; re-arm for the next
        self._op(f"operator {action}", lambda: "operator {action}: {device} written in {write_ms:.0f} ms".format(
            **self.app.plc.operator_write(action)))

    def pulse_x0(self):
        if not self.app.plc.simulator_mode:
            return

        def run():
            self.app.plc.simulator_test_write("X0", True)
            time.sleep(0.4)
            self.app.plc.simulator_test_write("X0", False)
        self._op("simulator X0 pulse", run)

    def send(self, verdict):
        trig = self.app.plc.current_trigger()
        if self.cmd_busy or trig is None or trig.state != "PENDING" or not self.arm.get():
            return
        self.cmd_busy = True
        self.arm.deselect()                              # one press = one command
        tid = trig.id

        def run():
            fn = self.app.plc.send_pass if verdict == "PASS" else self.app.plc.send_reject
            res = fn(tid)
            with self._lock:
                self.last_cmd = res
        threading.Thread(target=run, daemon=True).start()

    def close(self):
        self._closed = True
        if self.hmi is not None and self.hmi._alive:
            self.hmi.close()

    def refresh(self):
        pass

    # ------------------------------------------------------------ periodic view update (main thread)
    def _tick(self):
        if self._closed:
            return
        try:
            self._update()
        except Exception:                                # a display glitch must never stop the loop
            traceback.print_exc()
        self.app.after(200, self._tick)

    def _fmt_event(self, e):
        t = time.strftime("%H:%M:%S", time.localtime(e.wall))
        lab, val = None, ""
        if e.event == "DEVICE_CHANGE":
            lab, val = self.LOG_LABEL.get(e.device, e.device), e.ack
        elif e.event == "TRIGGER":
            lab, val = "INSPECTION", f"PENDING  (trigger #{e.trigger_id})" + ("  [after reconnect: may be stale]" if e.ack else "")
        elif e.event == "COMMAND_SENT":
            lab, val = "RESULT", f"{e.command}  (test command, trigger #{e.trigger_id})"
        elif e.event == "COMMAND_WRITTEN":
            lab, val = "PLC COMMAND", f"{e.device}  write ok {e.latency_ms:.0f} ms"
        elif e.event == "COMMAND_ACKED":
            lab, val = "PLC ACK", f"OK  {e.latency_ms:.0f} ms"
        elif e.event in ("COMMAND_NOT_ACKED", "COMMAND_ACK_LOST", "COMMAND_WRITE_FAILED", "COMMAND_REFUSED"):
            lab, val = ("PLC ACK" if "ACK" in e.event else "PLC COMMAND"), f"{e.event[8:]}  {e.error}"
        elif e.event == "SIM_TEST_WRITE":
            lab, val = "SIM TEST", e.ack.replace("SIMULATOR-ONLY stimulus: ", "")
        elif e.event in ("TRIGGER_CANCELLED", "TRIGGER_LOST", "TRIGGER_OVERDUE"):
            lab, val = "INSPECTION", f"{e.event[8:]}  {e.error}"
        elif e.event in ("CONNECTED", "FAULT", "CONNECT_FAILED", "DISCONNECTED"):
            lab, val = "PLC LINK", f"{e.event}  {e.error}"
        return f"{t}  {lab:<13} {val}" if lab else None

    def _update(self):
        svc = self.app.plc
        h = svc.health_check()
        st = h["link_state"]
        self.state.configure(text=st, text_color={PLC_CONNECTED: GOOD, PLC_DEGRADED: WARN, PLC_FAULT: BAD}.get(st, DIM))
        self.where_lbl.configure(text="SIMULATOR" if svc.simulator_mode else "REAL PLC",
                                 fg_color=ACC if svc.simulator_mode else BAD)
        snap = svc.snapshot() if st == PLC_CONNECTED else None
        if snap is not None and snap["age_s"] > 1.5:     # connected but the values are old: do not show them as live
            snap = None
        running = bool(snap and snap["plc_run"])
        self.run_lbl.configure(text="PLC --" if not snap else ("PLC RUN" if running else "PLC STOP"),
                               text_color=DIM if not snap else (GOOD if running else BAD))
        # ---- proof of life + what is actually answering
        now_m = time.monotonic()
        if now_m - self._ep[0] > 1.0:
            port = getattr(svc.client.transport, "port", None)
            self._ep = (now_m, plc_endpoint_info(port) if svc.simulator_mode and isinstance(port, int) else None)
        ep = self._ep[1]
        if not svc.simulator_mode:
            where = f"endpoint: {svc.client.transport.description} (physical link)"
        elif ep is None:
            where = "endpoint: simulator (process info unavailable)"
        else:
            where = ("endpoint: " + (f"{ep['listener']} is running" if ep["listener"] else "NOTHING is listening on this port")
                     + "   |   ISPSoft/COMMGR: "
                     + ("ONLINE on the simulator: " + ", ".join(ep["others"]) if ep["others"] else
                        "offline (not needed: the simulator keeps running the ladder until Simulation is stopped)"))
        age = "--" if h["age_s"] is None else f"{h['age_s']:.1f} s"
        self.live_lbl.configure(
            text=f"{'LIVE ' if snap else 'NO DATA'}  replies {h['ok']}   last reply {age} ago   {where}",
            text_color=DIM if snap else BAD)
        # ---- auto-reconnect: only re-opens the link; no command is ever replayed (PLCService guarantees that)
        if (st == PLC_FAULT and self._want_link and self.auto.get() and not self._op_busy
                and now_m - self._last_auto > 3.0):
            self._last_auto = now_m
            self._op("auto-reconnect", svc.connect)
        with self._lock:
            res, msg = self.last_cmd, self.msg
            if res is not None:
                self.last_cmd = None
        lat = "--" if h["last_latency_ms"] is None else f"{h['last_latency_ms']:.0f} ms"
        lw = "never" if not h["last_ok_wall"] else time.strftime("%H:%M:%S", time.localtime(h["last_ok_wall"]))
        bad = (not msg[1]) or bool(h["last_error"] and st != PLC_CONNECTED)
        line2 = msg[0] if (msg[0] and (not msg[1] or st == PLC_CONNECTED)) else (h["last_error"] if bad else "")
        self.info.configure(text=f"{h['transport']}   latency {lat}   last OK {lw}" + (f"\n{line2}" if line2 else ""),
                            text_color=BAD if bad else DIM)
        flat = {}
        if snap:
            for k in ("inputs", "internal", "outputs", "timers", "counters"):
                flat.update(snap[k])
        for n, w in self.cells.items():
            if n not in flat:
                w.configure(text="--", fg_color=OFF)
            elif n[0] == "C":
                w.configure(text=str(flat[n]), fg_color=GOOD if n == "C0" else BAD)
            elif n[0] == "T":
                w.configure(text=str(flat[n]), fg_color=ACC if flat[n] else OFF)
            else:
                w.configure(text="ON" if flat[n] else "OFF", fg_color=GOOD if flat[n] else OFF)
        # switches mirror what the PLC reports, so none can claim a state the PLC does not have
        live = svc.simulator_mode and st == PLC_CONNECTED and running    # no test input while the ladder is not running
        now = time.monotonic()
        for dev, (sw, var) in self.sw.items():
            if dev in flat and now > self._hold.get(dev, 0) and bool(flat[dev]) != var.get():
                var.set(bool(flat[dev]))
            sw.configure(state="normal" if live else "disabled")
        self.pulse_btn.configure(state="normal" if live else "disabled")
        trig = svc.current_trigger() if st == PLC_CONNECTED else None
        if res is not None:
            self.cmd_busy = False
            self._last_res = res
        r = self._last_res
        if not snap:
            self.flow_lbl.configure(text=f"No PLC data ({st})", text_color=DIM)
        elif not running:
            self.flow_lbl.configure(text="PLC is in STOP: the ladder is not executing. Values below are memory contents only; "
                                         "test inputs and commands are disabled.", text_color=BAD)
        elif trig is not None and trig.state == "PENDING":
            line = getattr(self.app, "tab_production", None)
            owner = ("handled by the inspection line (Production tab)" if line is not None and line.running
                     else "inspection line not running: AI result: none  -  waiting for a test PASS / REJECT")
            self.flow_lbl.configure(text=f"INSPECTION TRIGGER RECEIVED (M2 ON, trigger #{trig.id})  -  {owner}",
                                    text_color=WARN)
        else:
            self.flow_lbl.configure(text="Waiting for bottle sensor (X0 -> M2).   Conveyor "
                                         + ("RUNNING" if flat["Y1"] else "STOPPED")
                                         + f"   |   inspected {flat['C0'] + flat['C1']}:  PASS {flat['C0']}  REJECT {flat['C1']}"
                                         + ("   REJECT SOLENOID ON" if flat["Y0"] else ""),
                                    text_color=BAD if flat["Y0"] else DIM)
        can = (st == PLC_CONNECTED and running and trig is not None and trig.state == "PENDING"
               and bool(self.arm.get()) and not self.cmd_busy)
        for b in (self.btn_pass, self.btn_rej):
            b.configure(state="normal" if can else "disabled")
        op_ok = (svc.operator_controls and st == PLC_CONNECTED and running and bool(self.op_arm.get())
                 and not self._op_busy)
        for a, b in self.op_btns.items():
            idle = a != "TRIGGER" or trig is None
            b.configure(state="normal" if op_ok and idle else "disabled")
        self.op_note.configure(
            text=("off - set \"plc_operator_controls\": true in settings.json" if not svc.operator_controls else
                  "ladder needs M10 || X1 (start), M11 || X2 (stop).  REJECT fires the cylinder."),
            text_color=DIM if not svc.operator_controls else WARN)
        if r is not None:
            tail = f"ack {r.ack_ms:.0f} ms" if r.status == "ACKED" else r.detail[:70]
            self.cmd_lbl.configure(text=f"{r.command} #{r.trigger_id}: {r.status}  {tail}",
                                   text_color=GOOD if r.status == "ACKED" else BAD)
        evs = svc.events()
        if len(evs) != self._log_n:
            self._log_n = len(evs)
            lines = [x for x in (self._fmt_event(e) for e in evs[-400:]) if x]
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", "\n".join(reversed(lines[-120:])))      # newest first
            self.log.configure(state="disabled")


# ---------------------------------------------------------------- Production
class ProductionTab:
    """THE operator screen: machine state, both cameras live, the current bottle, counters, recent
    bottles and alarms, and three large controls (START INSPECTION / STOP / RESET FAULT).

    The logic lives elsewhere: machine_cycle.py (trigger -> per-bottle inspection -> FIFO -> PLC command),
    decision.py (the only place a PASS / REJECT / FAULT is decided), machine_state.py (the ONE machine
    state every widget shows), tracking.py (time-based bottle tracking, camera stations), alarms.py and
    production_store.py (coded alarms, persistent history + evidence). This tab gathers cached facts,
    shows them and starts/stops the line. It shares the app's one PLCService and CameraSet.

    Start sequence: readiness check (machine_state.readiness) -> load models (background) -> open the line
    cameras -> INITIALIZING until every camera has delivered a frame -> RUNNING. Nothing here starts on
    its own: the line, the cameras and the PLC link all need an explicit action.

    Engineer mode adds the line timing row, the speed-calibration and line-layout dialogs, the AI task,
    camera scan, the software HALT latch and the simulator bottle feed."""

    SLOTS = ("Camera 0", "Camera 1")
    TIMING = (("inspection_to_reject_mm", "Inspect->reject mm", 70), ("conveyor_mm_s", "Speed mm/s", 60),
              ("plc_t0_s", "T0 s", 46), ("plc_t1_s", "T1 s", 46), ("reject_tolerance_s", "Late limit s", 46),
              ("inspect_frames", "Frames", 36), ("inspect_window_s", "Window s", 46),
              ("capture_delay_s", "Capture delay s", 46))
    WARMUP_S = 8.0                               # line cameras must deliver a first frame within this

    def __init__(self, app: App, parent):
        self.app = app
        self.line = None                         # machine_cycle.MachineCycle (kept after Stop for display)
        self._imgs: dict = {}
        self._srcmap: dict = {}                  # option label -> camera index
        self._table_sig = None
        self._alarm_sig = None
        self._next_feed = 0.0
        self._tick_n = 0
        self._loading = False
        self._warming = None                     # (deadline, task, srcs, models, cams) while INITIALIZING
        self._stopping = False
        self._cam_test = False
        self._testing = False
        self._test_rec = None
        self._quality: dict = {}
        self._quality_at = 0.0
        self._cam_bad: dict = {}                 # slot -> last logged fault state
        self._reopen_at: dict = {}
        self._reopening: dict = {}
        self._reopens: dict = {}                 # slot -> reconnect attempts this session
        self.mstate, self.mreason = MS.OFFLINE, ""
        self.msg = ("", DIM)
        cfg = MC.line_settings(app.settings)

        # ---- state banner: the ONE machine state, the PLC mode, job / product
        ban = ctk.CTkFrame(parent, fg_color=PANEL, border_width=1, border_color=LINE)
        ban.pack(fill="x", pady=(0, 6))
        self.state_box = ctk.CTkFrame(ban, fg_color=OFF, corner_radius=6, width=290)
        self.state_box.pack(side="left", padx=8, pady=8, fill="y")
        self.line_lbl = ctk.CTkLabel(self.state_box, text=MS.OFFLINE, font=theme.BIG, text_color=ACC_T, width=270)
        self.line_lbl.pack(padx=10, pady=(6, 0))
        self.state_reason = ctk.CTkLabel(self.state_box, text="", font=theme.TINY, text_color=ACC_T, wraplength=270)
        self.state_reason.pack(padx=10, pady=(0, 6))
        info = ctk.CTkFrame(ban, fg_color="transparent")
        info.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        r1 = ctk.CTkFrame(info, fg_color="transparent")
        r1.pack(fill="x")
        self.plc_mode_lbl = ctk.CTkLabel(r1, text="", font=theme.CAPS, corner_radius=6, text_color=ACC_T,
                                         fg_color=OFF, width=120)
        self.plc_mode_lbl.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(r1, text="Job").pack(side="left", padx=(0, 4))
        self.job = ctk.CTkEntry(r1, width=130, placeholder_text="job / batch")
        if cfg.get("job_id"):
            self.job.insert(0, str(cfg["job_id"]))
        self.job.pack(side="left")
        ctk.CTkLabel(r1, text="Product").pack(side="left", padx=(10, 4))
        self.product = ctk.CTkEntry(r1, width=150, placeholder_text="e.g. 250 ml bottle")
        if cfg.get("product"):
            self.product.insert(0, str(cfg["product"]))
        self.product.pack(side="left")
        self.models_lbl = ctk.CTkLabel(info, text="", font=MONO, text_color=DIM, anchor="w", justify="left")
        self.models_lbl.pack(fill="x", pady=(4, 0))
        self.msg_lbl = ctk.CTkLabel(info, text="", font=theme.SMALL, anchor="w", justify="left", wraplength=900)
        self.msg_lbl.pack(fill="x")

        # ---- the three operator controls + tests
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 6))
        big = ("Segoe UI", 18, "bold")
        self.btn_start = ctk.CTkButton(bar, text="START INSPECTION", width=230, height=54, font=big, fg_color=GOOD,
                                       text_color=ACC_T, hover_color=GOOD, command=self.start_line)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ctk.CTkButton(bar, text="STOP", width=150, height=54, font=big, fg_color=BAD, text_color=ACC_T,
                                      hover_color=BAD, command=self.stop_line)
        self.btn_stop.pack(side="left", padx=6)
        self.btn_reset = ctk.CTkButton(bar, text="RESET FAULT", width=170, height=54, font=big, fg_color=WARN,
                                       text_color=ACC_T, hover_color=WARN, command=self.reset_halt)
        self.btn_reset.pack(side="left", padx=6)
        self.btn_test = ctk.CTkButton(bar, text="TEST INSPECTION\n(no PLC)", width=150, height=54,
                                      command=self.test_inspection)
        self.btn_test.pack(side="left", padx=(18, 6))
        self.btn_camtest = ctk.CTkButton(bar, text="CAMERA TEST", width=130, height=54, command=self.toggle_cam_test)
        self.btn_camtest.pack(side="left", padx=6)
        self.btn_snap = ctk.CTkButton(bar, text="Capture\ntest frame", width=100, height=54, command=self.capture_test)
        self.btn_snap.pack(side="left", padx=6)

        # ---- engineer row: task, cameras, simulator, timing, calibration (hidden for the operator)
        self.eng = ctk.CTkFrame(parent, fg_color=PANEL, border_width=1, border_color=WARN)
        eng1 = ctk.CTkFrame(self.eng, fg_color="transparent")
        eng1.pack(fill="x", padx=6, pady=(6, 2))
        ctk.CTkLabel(eng1, text="ENGINEER", font=theme.CAPS, text_color=WARN).pack(side="left", padx=(4, 10))
        ctk.CTkLabel(eng1, text="AI task").pack(side="left", padx=(0, 4))
        self.task = ctk.CTkOptionMenu(eng1, values=list(DEC.TASK_LABELS), width=200)
        self.task.set(next((k for k, v in DEC.TASK_LABELS.items() if v == cfg["line_task"]), "Detection"))
        self.task.pack(side="left")
        self.cam_pick = {}
        for slot in self.SLOTS:
            ctk.CTkLabel(eng1, text=slot).pack(side="left", padx=(10, 4))
            self.cam_pick[slot] = ctk.CTkOptionMenu(eng1, values=["- not used -"], width=210)
            self.cam_pick[slot].pack(side="left")
        ctk.CTkButton(eng1, text="Scan cameras", width=110, command=self.scan).pack(side="left", padx=8)
        ctk.CTkButton(eng1, text="HALT (latch)", width=100, fg_color="transparent", border_width=1, text_color=BAD,
                      border_color=BAD, command=self.halt_line).pack(side="right", padx=4)
        eng2 = ctk.CTkFrame(self.eng, fg_color="transparent")
        eng2.pack(fill="x", padx=6, pady=(0, 2))
        self.tim = {}
        for key, label, w in self.TIMING:
            ctk.CTkLabel(eng2, text=label, font=theme.SMALL).pack(side="left", padx=(6, 2))
            e = ctk.CTkEntry(eng2, width=w)
            e.insert(0, f"{cfg[key]:g}")
            e.pack(side="left")
            self.tim[key] = e
        ctk.CTkLabel(eng2, text="FAULT ->", font=theme.SMALL).pack(side="left", padx=(8, 2))
        self.fault_action = ctk.CTkOptionMenu(eng2, values=["REJECT", "PASS"], width=86)
        self.fault_action.set(str(cfg["fault_action"]).upper())
        self.fault_action.pack(side="left")
        ctk.CTkLabel(eng2, text="E-stop input", font=theme.SMALL).pack(side="left", padx=(8, 2))
        self.estop_dev = ctk.CTkEntry(eng2, width=46, placeholder_text="X3")
        if cfg["estop_device"]:
            self.estop_dev.insert(0, str(cfg["estop_device"]))
        self.estop_dev.pack(side="left")
        ctk.CTkButton(eng2, text="Save", width=60, command=self.save_timing).pack(side="left", padx=8)
        eng3 = ctk.CTkFrame(self.eng, fg_color="transparent")
        eng3.pack(fill="x", padx=6, pady=(0, 6))
        ctk.CTkButton(eng3, text="Speed calibration...", width=150,
                      command=lambda: hmi.SpeedCalibrationDialog(self.app, self._reload_timing)).pack(side="left", padx=4)
        ctk.CTkButton(eng3, text="Line layout / camera stations...", width=230,
                      command=self.open_layout).pack(side="left", padx=4)
        if app.settings.get("show_simulation_check", True):            # engineer-only; switch off when the system is proven
            ctk.CTkButton(eng3, text="Simulation check...", width=150, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                          command=lambda: hmi.SimulationCheckDialog(self.app)).pack(side="left", padx=4)
        ctk.CTkButton(eng3, text="Recipe...", width=90,
                      command=lambda: hmi.RecipeDialog(self.app, detect.CLASS_NAMES)).pack(side="left", padx=4)
        self.tim_msg = ctk.CTkLabel(eng3, text="", font=theme.SMALL, text_color=DIM, anchor="w")
        self.tim_msg.pack(side="left", padx=8)
        self.feed_s = ctk.CTkEntry(eng3, width=40)
        self.feed_s.insert(0, "4")
        self.feed_s.pack(side="right", padx=(2, 4))
        self.feed = ctk.CTkCheckBox(eng3, text="auto-feed every (s)", width=140)
        self.feed.pack(side="right")
        self.sim_btn = ctk.CTkButton(eng3, text="Simulate bottle (X0)", width=150, fg_color="transparent",
                                     border_width=1, text_color=WARN, command=self.simulate_bottle)
        self.sim_btn.pack(side="right", padx=6)

        # ---- centre: two live cameras + the current bottle
        self.mid = ctk.CTkFrame(parent, fg_color="transparent")
        self.mid.pack(fill="both", expand=True, pady=(0, 6))
        for c, w in enumerate((5, 5, 4)):
            self.mid.grid_columnconfigure(c, weight=w, uniform="m")
        self.mid.grid_rowconfigure(0, weight=1)
        self.cam_lbl, self.cam_img, self.cam_title = {}, {}, {}
        for i, slot in enumerate(self.SLOTS):
            f = ctk.CTkFrame(self.mid, fg_color=PANEL, border_width=1, border_color=LINE)
            f.grid(row=0, column=i, sticky="nsew", padx=(0, 6))
            self.cam_title[slot] = ctk.CTkLabel(f, text=f"CAMERA {i + 1}", font=theme.CAPS, text_color=DIM, anchor="w")
            self.cam_title[slot].pack(fill="x", padx=10, pady=(8, 2))
            # plain tk.Label: see LiveTab.build_panes (a CTkLabel re-scales its image on every frame)
            self.cam_img[slot] = tk.Label(f, text="camera off", fg=OFF, bg=VIDEO_BG, height=8, bd=0,
                                          compound="center", font=theme.BODY)
            self.cam_img[slot].pack(fill="both", expand=True, padx=10)
            self.cam_lbl[slot] = ctk.CTkLabel(f, text="", font=MONO, justify="left", anchor="w")
            self.cam_lbl[slot].pack(fill="x", padx=10, pady=(4, 8))
        ins = ctk.CTkFrame(self.mid, fg_color=PANEL, border_width=1, border_color=LINE)
        ins.grid(row=0, column=2, sticky="nsew")
        ctk.CTkLabel(ins, text="CURRENT BOTTLE", font=theme.CAPS, text_color=DIM, anchor="w").pack(fill="x", padx=10,
                                                                                                  pady=(8, 2))
        self.res_box = ctk.CTkFrame(ins, fg_color=PANEL_2, corner_radius=6)
        self.res_box.pack(fill="x", padx=10)
        self.res_lbl = ctk.CTkLabel(self.res_box, text="--", font=theme.HUGE, text_color=DIM)
        self.res_lbl.pack(pady=4)
        self.ins_lbl = ctk.CTkLabel(ins, text="", font=MONO, justify="left", anchor="nw", wraplength=420)
        self.ins_lbl.pack(fill="x", padx=10, pady=(6, 4))
        self.ev_img = ctk.CTkLabel(ins, text="no bottle yet", text_color=OFF, fg_color=VIDEO_BG, height=110,
                                   corner_radius=6)
        self.ev_img.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.ready_lbl = ctk.CTkLabel(ins, text="", font=theme.SMALL, justify="left", anchor="nw", wraplength=420)
        self.ready_lbl.pack(fill="x", padx=10, pady=(0, 8))
        self.plc_lbl = self.ready_lbl            # compatibility name (machine data is in the readiness list)

        # ---- counters
        prod = ctk.CTkFrame(parent, fg_color=PANEL, border_width=1, border_color=LINE)
        prod.pack(fill="x", pady=(0, 6), before=self.mid)       # counters always visible, above the cameras
        self.cnt = {}
        for key, title, col in (("total", "TOTAL", INK), (infer.PASS, "PASS", GOOD), (infer.REJECT, "REJECT", BAD),
                                (infer.FAULT, "FAULT", WARN), ("not_inspected", "NOT INSPECTED", WARN),
                                ("missed_reject", "MISSED REJECT", BAD), ("queue", "QUEUE", ACC)):
            cell = ctk.CTkFrame(prod, fg_color="transparent")
            cell.pack(side="left", padx=14, pady=6)
            self.cnt[key] = ctk.CTkLabel(cell, text="0", font=theme.BIG, text_color=col)
            self.cnt[key].pack()
            ctk.CTkLabel(cell, text=title, text_color=DIM, font=theme.CAPS).pack()
        self.travel_lbl = ctk.CTkLabel(prod, text="", font=MONO, text_color=DIM, justify="left")
        self.travel_lbl.pack(side="right", padx=10)

        # ---- recent bottles + alarms
        low = ctk.CTkFrame(parent, fg_color="transparent", height=170)
        low.pack(fill="x")
        self.table = ctk.CTkTextbox(low, font=MONO, wrap="none", height=150)
        self.table.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.table.insert("end", "RECENT BOTTLES\nnone yet - the list fills when the line runs")
        self.table.configure(state="disabled")
        self.alarms = ctk.CTkTextbox(low, font=("Consolas", 13), text_color=BAD, width=430, height=150)
        self.alarms.pack(side="left", fill="y")
        self.alarms.insert("end", "ALARMS\nshown here while the line runs; history on the History page")
        self.alarms.configure(state="disabled")
        self._shown_seq = {}
        app.after(300, self._tick)
        app.after(300, self._video_tick)

    # ------------------------------------------------------------ helpers
    @property
    def running(self) -> bool:
        return self.line is not None and self.line.running

    @property
    def busy(self) -> bool:
        """Line running, starting, or a camera / AI test holding the cameras."""
        return self.running or self._loading or self._warming is not None or self._cam_test or self._testing

    def set_mode(self, mode: str):
        if mode == "engineer":
            self.eng.pack(fill="x", pady=(0, 6), before=self.mid)
        else:
            self.eng.pack_forget()

    def refresh(self):
        if not self._srcmap and not self._loading:
            # Camera NAMES only (a COM query, no device is opened). "Scan cameras" probes the devices.
            try:
                names = infer.camera_names()
            except Exception:                                    # noqa: BLE001
                names = []
            if names:
                self._set_cams([{"index": i, "name": n, "width": 0, "height": 0} for i, n in enumerate(names)],
                               probed=False)

    def scan(self):
        if self.busy:
            return messagebox.showinfo("Cameras in use", "Stop the line / camera test before scanning for cameras.")
        n = int(self.app.settings.get("camera_probe", 5))
        self._loading = True
        self.app.run_bg(lambda: infer.list_cameras(n), self._set_cams)

    def _set_cams(self, cams, probed: bool = True):
        if probed:
            self._loading = False
        self._srcmap = {f"{c['index']}: {c.get('name') or 'camera'}"
                        + (f" ({c['width']}x{c['height']})" if c.get("width") else ""): c["index"] for c in cams}
        values = ["- not used -"] + list(self._srcmap)
        want = [int(s) for s in self.app.settings.get("line_cameras", []) if str(s).isdigit()]
        if not want:                                   # default: the EMEET Nova 4K units, else the first cameras
            want = [c["index"] for c in cams if "EMEET" in str(c.get("name", "")).upper()][:2]
        for slot, src in zip(self.SLOTS, want + [None] * 2):
            self.cam_pick[slot].configure(values=values)
            self.cam_pick[slot].set(next((k for k, v in self._srcmap.items() if v == src), values[0]))

    def chosen(self) -> list:
        out = []
        for slot in self.SLOTS:
            src = self._srcmap.get(self.cam_pick[slot].get())
            if src is not None and src not in out:
                out.append(src)
        return out

    def _timing_from_ui(self) -> dict:
        out = {}
        for key, label, _ in self.TIMING:
            try:
                v = float(self.tim[key].get())
            except ValueError:
                raise ValueError(f"{label}: not a number") from None
            if v < 0:
                raise ValueError(f"{label}: must be >= 0")
            out[key] = int(v) if key == "inspect_frames" else v
        if out["inspect_frames"] < 1:
            raise ValueError("Frames: at least 1")
        if out["plc_t0_s"] <= 0 or out["plc_t1_s"] <= 0:
            raise ValueError("T0 and T1 must be > 0 (they are the ladder's timer presets)")
        out["fault_action"] = self.fault_action.get()
        dev = self.estop_dev.get().strip().upper()
        if dev:
            try:
                PLC_AM.parse(dev)
            except ValueError as e:
                raise ValueError(f"E-stop input: {e}") from None
        out["estop_device"] = dev
        return out

    def save_timing(self) -> bool:
        try:
            vals = self._timing_from_ui()
        except ValueError as e:
            self.tim_msg.configure(text=str(e), text_color=BAD)
            return False
        vals["job_id"], vals["product"] = self.job.get().strip(), self.product.get().strip()
        self.app.settings.update(vals)
        D.save_settings(self.app.settings)
        cfg = MC.line_settings(self.app.settings)
        bad = MC.line_problems(cfg, self.chosen())
        if bad:
            self.tim_msg.configure(text=f"saved, but the line will not start: {'; '.join(bad)}"[:300], text_color=BAD)
            return False
        tt, measured = MC.travel_time(cfg)
        self.tim_msg.configure(text=f"saved: travel {tt:.2f} s" + ("" if measured else " (= T0: distance/speed not set)")
                               + f"  |  {TR.position_source(cfg).describe()}", text_color=DIM)
        return True

    def _reload_timing(self):
        """A dialog saved settings: show them in the timing row."""
        cfg = MC.line_settings(self.app.settings)
        for key, _, _ in self.TIMING:
            self.tim[key].delete(0, "end")
            self.tim[key].insert(0, f"{cfg[key]:g}")
        self.save_timing()

    def open_layout(self):
        names = {v: k for k, v in self._srcmap.items()}
        hmi.LineLayoutDialog(self.app, [(s, names.get(s, f"camera {s}")) for s in self.chosen()], self._reload_timing)

    # ------------------------------------------------------------ machine state (the one source of truth)
    def facts(self, models_injected: bool = False) -> dict:
        svc = self.app.plc
        h = svc.health_check()
        link = h["link_state"]
        snap = svc.snapshot() if link == PLC_CONNECTED else None
        if snap is not None and snap["age_s"] > 1.5:
            snap = None
        timing = []
        try:                                       # what is typed in the timing row is what Start will save
            cfg = MC.line_settings(dict(self.app.settings, **self._timing_from_ui()))
        except ValueError as e:
            cfg = MC.line_settings(self.app.settings)
            timing.append(str(e))
        timing += MC.line_problems(cfg, self.chosen())
        task = DEC.TASK_LABELS.get(self.task.get(), "detection")
        stages = DEC.TASKS.get(task, ())
        cams = []
        for slot in self.SLOTS:
            pick = self.cam_pick[slot].get()
            if pick == "- not used -":
                continue
            src = self._srcmap.get(pick)
            cam = self.app.cams.get(src) if src is not None else None
            if self.running and cam is not None:
                ok = bool(cam.alive and not cam.error)
                cams.append((slot, ok, "streaming" if ok else (cam.error or "not streaming")))
            else:
                cams.append((slot, src is not None, "selected" if src is not None else "not found"))
        models = []
        if self.line is not None and self.running:
            mids = self.line.inspector.models()
            models = [(st, True, str(mids.get(st))) for st in stages]
        else:
            for st in stages:
                if models_injected:
                    models.append((st, True, "provided"))
                elif st == DEC.CLASSIFICATION:
                    stamp = D.load_config().get("active_model")
                    ok = bool(stamp) and (D.MODELS / str(stamp) / "model.pt").exists()
                    models.append((st, ok, str(stamp) if ok else "no active classifier"))
                elif st == DEC.DETECTION:
                    w = Path(self.app.settings.get("detector_weights") or detect.DEFAULT_WEIGHTS)
                    models.append((st, w.exists(), w.name if w.exists() else f"{w.name} missing"))
                else:
                    w = self.app.settings.get("segmenter_weights")
                    models.append((st, bool(w) and Path(w).exists(), "no segmentation model trained"))
        dev = str(cfg.get("estop_device") or "").upper()
        estop = None
        if dev and snap is not None:
            flat = {}
            for k in ("inputs", "internal", "outputs"):
                flat.update(snap[k])
            if dev in flat:
                v = bool(flat[dev])
                estop = v if cfg.get("estop_active_high") else not v
        recipe = []
        if DEC.DETECTION in stages:
            r = D.load_config().get("inspection")
            recipe = DEC.recipe_problems(r) if r else []
        line = self.line if self.running else None
        halted = line.halted if line is not None else None
        return {"plc_link": link, "plc_run": None if snap is None else bool(snap["plc_run"]),
                "plc_real": not svc.simulator_mode, "cameras": cams, "models": models,
                "timing": timing, "estop": estop, "recipe": recipe,
                "line_running": self.running, "line_starting": self._loading or self._warming is not None,
                "line_stopping": self._stopping, "line_halted": halted,
                "halted_by_estop": bool(halted and "E-stop" in halted),
                "line_error": line.error if line is not None else None,
                "inspecting": line.snapshot()["inspecting"] if line is not None else 0,
                "starting_detail": "opening cameras, waiting for first frames" if self._warming else "loading models"}

    # ------------------------------------------------------------ line control
    def start_line(self, models: dict | None = None):
        """models: injected {"classification"/"detection"/"segmentation": object} for the self-test;
        normally the required models are loaded in the background first."""
        if self.running or self._loading or self._warming is not None:
            return
        if self._cam_test:
            self.toggle_cam_test()
        if self.app.plc.link_state() != PLC_CONNECTED:
            return messagebox.showwarning("PLC", "Connect the PLC first (Machine tab): the line is driven by its "
                                                 "trigger (X0 -> M2).")
        srcs = self.chosen()
        if not srcs:
            return messagebox.showinfo("No camera", "Choose Camera 0 and/or Camera 1 (engineer mode), then Start.")
        if not self.save_timing():
            return messagebox.showwarning("Timing", self.tim_msg.cget("text"))
        st, why = MS.state(self.facts(models_injected=models is not None))
        if st != MS.READY:
            bad = [c for c in MS.readiness(self.facts(models_injected=models is not None)) if not c.ok]
            return messagebox.showwarning("Not ready", "The line cannot start:\n\n" + "\n".join(
                f"- {c.name}: {c.detail}  ->  {c.action}" for c in bad) or why)
        task = DEC.TASK_LABELS[self.task.get()]
        self.app.settings.update(line_task=task, line_cameras=srcs)
        D.save_settings(self.app.settings)
        stages = DEC.TASKS[task]
        if models is not None:
            return self._go(task, srcs, models)
        self._loading = True

        def work():
            out, errs = {}, []
            if DEC.CLASSIFICATION in stages:
                stamp = D.load_config().get("active_model")
                if not stamp:
                    errs.append("Classification: no active classifier in this project (Models -> activate)")
                else:
                    try:
                        out[DEC.CLASSIFICATION] = infer.Model(stamp)
                    except Exception as e:                   # noqa: BLE001 - shown to the operator
                        errs.append(f"Classification: {type(e).__name__}: {e}")
            if DEC.DETECTION in stages:
                try:
                    out[DEC.DETECTION] = self.app.tab_live.build_detector()
                except Exception as e:                       # noqa: BLE001
                    errs.append(f"Detection: {e}")
            if DEC.SEGMENTATION in stages:
                try:
                    out[DEC.SEGMENTATION] = segment.YoloSegmenter(
                        weights=self.app.settings.get("segmenter_weights") or None,
                        conf=float(self.app.settings.get("segmenter_conf", segment.DEV_CONF)))
                except Exception as e:                       # noqa: BLE001
                    errs.append(f"Segmentation: {e}")
            return out, errs

        def done(r):
            self._loading = False
            out, errs = r
            if errs:
                for e in errs:
                    self.app.alarms.raise_("MODEL_NOT_FOUND", e, key=e[:20], source="start")
                return messagebox.showerror("Cannot start the line", "\n\n".join(errs))
            self._go(task, srcs, out)
        self.app.run_bg(work, done)

    def _open_cams(self, srcs) -> list:
        """Line cameras, capture only (models run per bottle in the Inspector). Stops the Live tab first."""
        if self.app.tab_live.running:
            self.app.tab_live.stop()
        cfg = MC.line_settings(self.app.settings)
        cams = []
        for slot, src in zip(self.SLOTS, srcs):
            cam = self.app.cams.add(src, f"{slot} (#{src})")
            cam.stop()                                       # a capture mode only applies on (re)open
            cam.capture_wh = tuple(cfg["line_capture_wh"]) if cfg["line_capture_wh"] else None
            cam.fourcc = cfg["line_fourcc"] or None
            cams.append(cam)
        self.app.cams.start(srcs, None, None)
        self._slot_src = dict(zip(self.SLOTS, srcs))
        return cams

    def _go(self, task, srcs, models):
        cams = self._open_cams(srcs)
        self._warming = (time.monotonic() + self.WARMUP_S, task, srcs, models, cams)
        self._check_warm()

    def _check_warm(self):
        """INITIALIZING: the line starts only when every line camera has delivered a frame."""
        if self._warming is None:
            return
        deadline, task, srcs, models, cams = self._warming
        if all(c.latest_frame() is not None for c in cams):
            self._warming = None
            return self._launch(task, models, cams)
        if time.monotonic() > deadline or any(c.error for c in cams):
            self._warming = None
            dead = [c.name for c in cams if c.latest_frame() is None]
            why = "; ".join(f"{c.name}: {c.error}" for c in cams if c.error) or f"no frame from {', '.join(dead)}"
            self.app.alarms.raise_("CAMERA_DISCONNECTED", f"line not started: {why}", source="start")
            self.app.cams.stop()
            self.msg = (f"Line NOT started: camera did not deliver images ({why})", BAD)

    def _launch(self, task, models, cams):
        cfg = MC.line_settings(self.app.settings)
        insp = MC.Inspector(cams, task, classifier=models.get(DEC.CLASSIFICATION),
                            detector=models.get(DEC.DETECTION), segmenter=models.get(DEC.SEGMENTATION),
                            frames=cfg["inspect_frames"], window_s=cfg["inspect_window_s"])
        svc = self.app.plc
        try:
            store = self.app.store()
            store.run_id = None                               # a new run per start
            line = MC.MachineCycle(svc, insp, cfg, log_dir=store.folder, alarms=self.app.alarms, store=store,
                                   run_info={"project": D.PROJECT, "recipe": D.load_config().get("inspection"),
                                             "mode": "SIMULATOR" if svc.simulator_mode else "REAL PLC"})
        except ValueError as e:
            self.app.cams.stop()
            self.msg = (f"Line NOT started: {e}", BAD)
            self.app.alarms.raise_("TIMING_INVALID", str(e), source="start")
            return
        svc.watch_x0 = True                                   # bottle accounting: X0 edges with no trigger
        self.line = line
        self.line.start()
        self.msg = (f"Line started ({'SIMULATOR' if svc.simulator_mode else 'REAL PLC'}), run {store.run_id}", DIM)
        self._table_sig = None

    def stop_line(self):
        """Orderly STOP: no new triggers are answered, bottles still being inspected end as FAULT (remove by
        hand), cameras stop. Bottles already answered are left to the PLC."""
        self._stopping = True
        try:
            self._warming = None
            if self.line is not None and self.line.running:
                self.line.stop()
                self.msg = ("Line stopped by the operator.", DIM)
            self.app.plc.watch_x0 = False
            self.app.cams.stop()
            self._cam_test = False
            self.btn_camtest.configure(text="CAMERA TEST")
        finally:
            self._stopping = False

    def halt_line(self):
        """Software HALT latch: no PLC command is sent until RESET FAULT. The hardware E-stop is what actually
        removes power; this stops the program answering the PLC."""
        if self.line is not None and self.line.running:
            self.line.halt("operator HALT button")

    def reset_halt(self):
        if self.line is not None and self.line.halted:
            if not self.line.reset():
                return messagebox.showwarning("Cannot reset", "The hardware E-stop input still reads pressed.")
        else:
            self.app.alarms.acknowledge_all()
        self.msg = ("Fault reset / alarms acknowledged.", DIM)

    def close(self):
        self.stop_line()

    def simulate_bottle(self):
        """Simulator only: pulse X0 (the photo-eye) through the Machine tab's guarded test stimulus."""
        if not self.app.plc.simulator_mode:
            return messagebox.showinfo("Simulator only", "Bottles can only be simulated on the ISPSoft simulator.")
        self.app.tab_machine.pulse_x0()

    # ------------------------------------------------------------ tests without the PLC
    def toggle_cam_test(self):
        """CAMERA TEST: open the chosen cameras (capture only) and show them live with image quality, no PLC,
        no AI. Press again to stop."""
        if self._cam_test:
            self.app.cams.stop()
            self._cam_test = False
            self.btn_camtest.configure(text="CAMERA TEST")
            return
        if self.running or self._warming is not None or self._loading:
            return messagebox.showinfo("Line running", "Stop the line first.")
        srcs = self.chosen()
        if not srcs:
            return messagebox.showinfo("No camera", "Choose the line cameras first (engineer mode).")
        self._open_cams(srcs)
        self._cam_test = True
        self.btn_camtest.configure(text="STOP CAMERA TEST")
        self.msg = ("Camera test: live images, no PLC, no AI.", DIM)

    def capture_test(self):
        """Save the newest frame of every streaming line camera to captures/camtest_<time>/."""
        cams = [self.app.cams.get(s) for s in getattr(self, "_slot_src", {}).values()]
        frames = [(c, c.latest_frame()) for c in cams if c is not None]
        frames = [(c, f) for c, f in frames if f is not None]
        if not frames:
            return messagebox.showinfo("Capture", "No camera is streaming: start the CAMERA TEST first.")
        out = Path("captures") / f"camtest_{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        for c, f in frames:
            cv2.imwrite(str(out / f"cam{c.source}_seq{f.seq}.png"), f.image)
        q = {str(c.source): bench.frame_quality(f.image) for c, f in frames}
        (out / "quality.json").write_text(json.dumps(q, indent=2))
        self.msg = (f"Saved {len(frames)} test frame(s) to {out}", DIM)

    def test_inspection(self):
        """TEST INSPECTION: what would the line decide about the bottle in front of the cameras now? Same models,
        decision engine and recipe as production; the PLC is never touched. Result + images in captures/."""
        if self.running or self._warming is not None or self._loading or self._testing:
            return messagebox.showinfo("Busy", "Stop the line first.")
        srcs = self.chosen()
        if not srcs:
            return messagebox.showinfo("No camera", "Choose the line cameras first (engineer mode).")
        if self._cam_test:
            self.toggle_cam_test()
        if self.app.tab_live.running:
            self.app.tab_live.stop()
        self.app.cams.stop()
        task = DEC.TASK_LABELS[self.task.get()]
        self._testing = True
        self.res_lbl.configure(text="TESTING...", text_color=WARN)
        self.ins_lbl.configure(text=f"{len(srcs)} camera(s), task {task}\n(the PLC is not used)")

        def done(rec):
            self._testing = False
            self._test_rec = rec
            dec = rec["decision"]
            col = {infer.PASS: GOOD, infer.REJECT: BAD}.get(dec["state"], WARN)
            self.res_lbl.configure(text=VD.shown_result(dec["state"]), text_color=col)
            self.res_box.configure(fg_color={infer.PASS: PASS_SOFT, infer.REJECT: REJECT_SOFT}.get(dec["state"],
                                                                                                  FAULT_SOFT))
            ms = [fr.get("ms", 0) for c in rec["cameras"].values() for fr in c.get("frames", [])]
            self.ins_lbl.configure(text=(f"TEST (no PLC)  {time.strftime('%H:%M:%S')}\n"
                                         f"Defect    {', '.join(VD.pretty(d) for d in dec['defects']) or '-'}\nReason    {dec['reason'][:160]}\n"
                                         f"AI time   {sum(ms):.0f} ms over {len(ms)} frame(s)\n"
                                         f"Saved     {rec.get('out_dir', '')}"))
            ov = sorted(Path(rec.get("out_dir", ".")).glob("*_overlay.jpg"))
            if ov:
                img = cv2.imread(str(ov[0]))
                if img is not None:
                    self._ev = bgr_to_ctk(img, (420, 260))
                    self.ev_img.configure(image=self._ev, text="")

        def work():
            try:
                return MC.bench_inspect("TEST", [str(s) for s in srcs], task=task, frames=2)
            finally:
                self._testing = False
        self.app.run_bg(work, done)

    # ------------------------------------------------------------ view (main thread)
    def _tick(self):
        try:
            self._check_warm()
            self._update()
        except Exception:                                    # a display glitch must never stop the loop
            traceback.print_exc()
        self.app.after(250, self._tick)

    def _video_tick(self):
        """Smooth camera preview (~25 fps), separate from the 250 ms status refresh. Only redraws a slot
        when its frame seq changed; downscales with cv2 first (PIL LANCZOS on a full frame is slow)."""
        t0 = time.monotonic()
        try:
            if self.running or self._cam_test or self._warming is not None:
                for slot in self.SLOTS:
                    src = getattr(self, "_slot_src", {}).get(slot)
                    cam = self.app.cams.get(src) if src is not None else None
                    if cam is None or not cam.armed:
                        continue
                    f = cam.latest_frame()
                    if f is None or self._shown_seq.get(slot) == (cam.session, f.seq):
                        continue
                    self._shown_seq[slot] = (cam.session, f.seq)
                    lab = self.cam_img[slot]
                    bw, bh = max(200, lab.winfo_width() - 4), max(150, lab.winfo_height() - 4)
                    h, w = f.image.shape[:2]
                    k = min(bw / w, bh / h)
                    small = cv2.resize(f.image, (max(1, int(w * k)), max(1, int(h * k))),
                                       interpolation=cv2.INTER_AREA)
                    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                    self._imgs[slot] = img
                    lab.configure(image=img, text="")
        except Exception:                                    # a display glitch must never stop the loop
            traceback.print_exc()
        self.app.after(max(10, 40 - int((time.monotonic() - t0) * 1000)), self._video_tick)

    def _show_state(self, f):
        st, why = MS.state(f)
        self.mstate, self.mreason = st, why
        cls = MS.LOOK[st][0]
        colour = {"PASS": GOOD, "BAD": BAD, "WARN": WARN, "ACC": ACC}.get(cls, OFF)
        self.state_box.configure(fg_color=colour)
        self.line_lbl.configure(text=st.replace("_", " "))
        self.state_reason.configure(text=(MS.LOOK[st][1] if st in (MS.READY, MS.RUNNING) else why)[:150])
        live = self.running or self._warming is not None or self._loading
        self.btn_start.configure(state="normal" if st == MS.READY else "disabled",
                                 fg_color=GOOD if st == MS.READY else PANEL_2)
        self.btn_stop.configure(state="normal" if live or self._cam_test else "disabled",
                                fg_color=BAD if live or self._cam_test else PANEL_2)
        alarm = bool(self.app.alarms.active()) or (self.line is not None and self.line.halted)
        self.btn_reset.configure(state="normal" if alarm else "disabled", fg_color=WARN if alarm else PANEL_2)
        idle = not live and not self._testing
        self.btn_test.configure(state="normal" if idle else "disabled")
        self.btn_camtest.configure(state="normal" if idle or self._cam_test else "disabled")
        # the start checklist (or, while running, what the machine is doing)
        if not live:
            lines = [f"{'OK ' if c.ok else 'NO '} {c.name}: {c.detail}"[:70] for c in MS.readiness(f)]
            self.ready_lbl.configure(text="START CHECKS\n" + "\n".join(lines),
                                     text_color=INK if st == MS.READY else WARN)
        else:
            self.ready_lbl.configure(text="", text_color=DIM)

    RECONNECT_S = 3.0                            # a dead line camera is reopened at most this often

    def _camera_watch(self, slot, cam, src):
        """Log camera state changes (events, not frames) and, while the line runs, reopen a dead camera in the
        background. Bottles inspected while it is down are FAULT (camera fault); nothing else is paused."""
        bad = bool(cam.error) or not cam.alive
        if bad != self._cam_bad.get(slot):
            self._cam_bad[slot] = bad
            applog.log("camera", f"{slot} (#{src}) " + (f"FAULT: {cam.error or 'not streaming'}" if bad else "streaming"),
                       "ERROR" if bad else "INFO", wh="x".join(map(str, cam.frame_wh or ())) or "-",
                       fps=round(cam.fps, 1))
        if (bad and self.running and not self._reopening.get(slot)
                and time.monotonic() - self._reopen_at.get(slot, 0.0) > self.RECONNECT_S):
            self._reopen_at[slot] = time.monotonic()
            self._reopening[slot] = True
            n = self._reopens[slot] = self._reopens.get(slot, 0) + 1
            applog.log("camera", f"{slot} (#{src}) reconnect attempt {n}", "WARNING")

            def work():
                try:
                    cam.stop()
                    if self.running:
                        cam.start(src)
                finally:
                    self._reopening[slot] = False
            threading.Thread(target=work, name=f"reopen-{slot}", daemon=True).start()

    def _monitor(self, f):
        """Condition alarms while the line is (starting to) run. Idle conditions are cleared: an operator who has
        not connected the PLC yet is not an alarm."""
        on = self.running or self._warming is not None
        A = self.app.alarms
        A.set_condition("PLC_COMM_FAULT", on and f["plc_link"] in (PLC_FAULT, "DISCONNECTED"),
                        f"link {f['plc_link']}")
        A.set_condition("PLC_NOT_RUNNING", on and f["plc_run"] is False, "PLC reports STOP")
        A.set_condition("E_STOP", f["estop"] is True, f"input {self.app.settings.get('estop_device')} pressed")

    def _update(self):
        self._tick_n += 1
        svc = self.app.plc
        f = self.facts()
        self._show_state(f)
        self._monitor(f)
        real = not svc.simulator_mode
        self.plc_mode_lbl.configure(text="REAL PLC" if real else "SIMULATOR", fg_color=BAD if real else ACC)
        self.sim_btn.configure(state="normal" if svc.simulator_mode and f["plc_run"] else "disabled")
        self.msg_lbl.configure(text=self.msg[0], text_color=self.msg[1])

        # cameras
        now = time.monotonic()
        cfg = MC.line_settings(self.app.settings)
        stations = TR.stations(cfg, self.chosen())
        want = tuple(cfg["line_capture_wh"]) if cfg["line_capture_wh"] else None
        showing = self.running or self._cam_test or self._warming is not None
        for i, slot in enumerate(self.SLOTS):
            src = getattr(self, "_slot_src", {}).get(slot) if showing else None
            cam = self.app.cams.get(src) if src is not None else None
            pick_src = self._srcmap.get(self.cam_pick[slot].get())
            stn = stations.get(str(pick_src)) if pick_src is not None else None
            title = f"CAMERA {i + 1}" + (f"  -  {stn.name}, {stn.role}, +{stn.offset_mm:g} mm"
                                         + (f", {stn.side}" if stn.side else "") if stn else "  -  not used")
            self.cam_title[slot].configure(text=title.upper())
            if cam is None or not cam.armed:
                self.cam_lbl[slot].configure(text="OFF" if pick_src is not None else "not used", text_color=DIM)
                if not showing:
                    self.cam_img[slot].configure(image="", text="camera off")
                continue
            self._camera_watch(slot, cam, src)
            age = None if cam.frame_ts is None else now - cam.frame_ts
            bandwidth = False
            if cam.error:
                state, col = f"FAULT: {cam.error}"[:60], BAD
                if "stopped returning" in cam.error and len(self.app.cams.running()) >= 2:
                    # measured 2026-10-03: two EMEET Nova 4K on one USB 2.0 hub -> only one high-res stream
                    state += "\nUSB BANDWIDTH? use separate USB 3 root ports"
                    bandwidth = True
            elif not cam.alive:
                state, col = "DISCONNECTED", BAD
            elif age is None:
                state, col = "CONNECTING...", WARN
            elif age > infer.MAX_RESULT_AGE_S:
                state, col = f"STALE: last frame {age:.1f} s ago", BAD
            else:
                state, col = "CONNECTED", GOOD
            if want and cam.frame_wh and tuple(cam.frame_wh) != want and cam.alive and not cam.error:
                state += f"\nrequested {want[0]}x{want[1]}, camera gives {cam.frame_wh[0]}x{cam.frame_wh[1]}"
                col = WARN if col == GOOD else col
                bandwidth = bandwidth or len(self.app.cams.running()) >= 2
            run = self.running
            self.app.alarms.set_condition("CAMERA_DISCONNECTED", run and (bool(cam.error) or not cam.alive),
                                          state, key=slot)
            self.app.alarms.set_condition("CAMERA_STALE", run and age is not None and age > infer.MAX_RESULT_AGE_S,
                                          state, key=slot)
            self.app.alarms.set_condition("CAMERA_BANDWIDTH_PROBLEM", bandwidth, state, key=slot)
            wh = "x".join(map(str, cam.frame_wh)) if cam.frame_wh else "--"
            ts = "--" if age is None else f"#{cam.frame_seq}  {age * 1000:.0f} ms ago"
            q = ""
            if self._cam_test and now - self._quality_at > 1.0:
                fr = cam.latest_frame()
                if fr is not None:
                    self._quality[slot] = bench.frame_quality(fr.image)
            if self._cam_test and slot in self._quality:
                qq = self._quality[slot]
                q = (f"\nbright {qq['brightness']:.0f}  contrast {qq['contrast']:.0f}  sharp {qq['sharpness']:.0f}  "
                     f"clipped {qq['clipped']:.1f}%")
            cur = ""
            if self.line is not None and self.running:
                for b in self.line.snapshot()["fifo"]:
                    if b.status == MC.INSPECTING:
                        cur = f"   bottle {b.inspection_id}"
                        break
            rc = f"  reconnects {self._reopens[slot]}" if self._reopens.get(slot) else ""
            self.cam_lbl[slot].configure(text=f"{state}\n{wh}  {cam.fps:.1f} fps  frame {ts}{cur}{rc}{q}", text_color=col)
        if self._cam_test and now - self._quality_at > 1.0:
            self._quality_at = now

        line = self.line
        if line is None:
            self.models_lbl.configure(text=f"task {self.task.get()}   recipe "
                                           f"{production_store.recipe_id(D.load_config().get('inspection'))}   "
                                           f"{TR.position_source(cfg).describe()}")
            return
        s = line.snapshot()
        models = line.inspector.models()
        self.models_lbl.configure(
            text=f"task {line.inspector.task}   classifier {models['classification'] or '-'}   detector "
                 f"{models['detection'] or '-'}   run {getattr(line.store, 'run_id', None) or '-'}   "
                 f"{s['position']}", text_color=DIM)
        c = s["counts"]
        for k, w in self.cnt.items():
            w.configure(text=str(s["queue"] if k == "queue" else c.get(k, 0)))
        p50, p95 = s["cycle_p50_ms"], s["cycle_p95_ms"]
        self.travel_lbl.configure(text=f"travel {s['travel_s']:.2f} s "
                                       + ("(measured)" if s["travel_measured"] else "(= T0: not measured)")
                                       + (f"\ntrigger->decision p50 {p50:.0f} ms  p95 {p95:.0f} ms" if p50 else ""))
        b = s["last"]
        if b is not None:
            shown = b.final or b.decision or "--"
            col = {infer.PASS: GOOD, infer.REJECT: BAD}.get(shown, WARN)
            self.res_lbl.configure(text=VD.shown_result(shown) + ("" if b.final else " ..."), text_color=col)
            self.res_box.configure(fg_color={infer.PASS: PASS_SOFT, infer.REJECT: REJECT_SOFT}.get(shown, FAULT_SOFT))
            conf = "--" if b.confidence is None else f"{b.confidence:.2f}"
            percam = "  ".join(f"{k}:{v[0]}" for k, v in b.per_camera.items())
            t = b.timings
            tim = "  ".join(f"{k.replace('_ms', '')} {t[k]:.0f}" for k in ("capture_wait_ms", "classification_ms",
                                                                            "detection_ms", "decision_ms", "plc_ms")
                            if t.get(k) is not None)
            self.ins_lbl.configure(text=(f"Bottle    {b.inspection_id}   (trigger #{b.trigger_id})\n"
                                         f"Defect    {', '.join(VD.pretty(d) for d in b.defects) or ('-' if b.decision == infer.PASS else b.reason)[:120]}\n"
                                         f"Conf.     {conf}\nCameras   {percam or '-'}\n"
                                         f"Command   {b.command or '-'}   PLC {b.plc_status or b.status}\n"
                                         f"ms        {tim or '-'}\n{b.note}"))
            if b.thumb is not None and getattr(self, "_thumb_for", None) is not b:
                self._thumb_for = b
                self._ev = bgr_to_ctk(b.thumb, (420, 260))
                self.ev_img.configure(image=self._ev, text="")
        # recent bottles
        rows = list(s["fifo"]) + list(reversed(s["history"]))
        sig = tuple((r.inspection_id, r.status, r.final, r.plc_status) for r in rows[:40])
        if sig != self._table_sig:
            self._table_sig = sig
            head = (f"{'ID':<7}{'TIME':<14}{'RESULT':<8}{'AI':<8}{'DEFECT':<26}{'CONF':>5}  {'SCHED.REJECT':<14}"
                    f"{'CMD':<7}{'STATUS':<14}PLC")
            lines = [head, "-" * len(head)]
            for r in rows[:200]:
                conf = "" if r.confidence is None else f"{r.confidence:.2f}"
                lines.append(f"{r.inspection_id:<7}{MC._wall(r.wall):<14}{(r.final or '...'):<8}{(r.decision or '...'):<8}"
                             f"{(', '.join(r.defects) or ('-' if r.decision == infer.PASS else r.reason))[:25]:<26}"
                             f"{conf:>5}  {MC._wall(r.scheduled_wall):<14}{(r.command or '-'):<7}{r.status:<14}"
                             f"{r.plc_status}")
            self.table.configure(state="normal")
            self.table.delete("1.0", "end")
            self.table.insert("end", "\n".join(lines))
            self.table.configure(state="disabled")
        # alarms (coded, active first) + the line's own event lines
        act = self.app.alarms.active()
        asig = (tuple((a.code, a.key, a.state, a.count) for a in act), len(s["alarms"]))
        if asig != self._alarm_sig:
            self._alarm_sig = asig
            txt = ["ACTIVE ALARMS" if act else "NO ACTIVE ALARM"]
            for a in act[:12]:
                txt.append(f"{time.strftime('%H:%M:%S', time.localtime(a.raised))} {a.severity[:4]} {a.code}"
                           + (f" x{a.count}" if a.count > 1 else "") + (" (ack)" if a.state == AL.ACKNOWLEDGED else "")
                           + f"\n    {a.message} -> {a.action}")
            txt.append("\nEVENTS (newest first)")
            txt += [f"{time.strftime('%H:%M:%S', time.localtime(t))}  {e}" for t, e in reversed(s["alarms"])]
            self.alarms.configure(state="normal")
            self.alarms.delete("1.0", "end")
            self.alarms.insert("end", "\n".join(txt))
            self.alarms.configure(state="disabled")
        # simulator auto-feed: one simulated bottle every N s while the line runs
        if self.feed.get() and self.running and svc.simulator_mode and f["plc_run"]:
            try:
                every = max(0.5, float(self.feed_s.get()))
            except ValueError:
                every = 4.0
            if now >= self._next_feed:
                self._next_feed = now + every
                self.app.tab_machine.pulse_x0()


# --------------------------------------------------------------------- Label
class LabelTab:
    """Thumbnail grid + one-image inspector for multi-label review.

    The selection is always the target: bulk buttons and keys act on it. In
    review mode (inspector open) clicking or stepping makes the inspected image
    the selection, so a key press labels exactly the image on screen.
    Edits are applied to app.labels at once (the next key press sees them) and
    written to labels.csv by ONE worker thread in order -- two racing writers
    could otherwise each load the CSV and drop the other's edit.
    """

    # No "+ve/-ve" here either: at dataset level it means pass/fail, at defect
    # level it means has/has-not, and one label for both reads as the wrong one.
    MODES = {"All images": "all",
             "Inbox - not reviewed": "inbox",
             "AI suggested - not reviewed": "suggested",
             "GOOD - passes": "good",
             "DEFECTIVE - any defect": "defective",
             "WITH this defect": "pos",
             "WITHOUT this defect": "neg"}
    LOW_CLASS = 50              # fewer training images than this is flagged on the balance strip
    CONFIDENT = 0.95            # "accept all confident" needs every probability >= this or <= 1 - this

    def __init__(self, app: App, parent):
        self.app, self.page, self.sel, self.items = app, 0, set(), []
        self.cards: dict = {}
        self._imgs: list = []          # Tk drops images that nothing references
        self._token = 0                # ignore thumbnails from a superseded page
        self.last_click = None         # anchor for shift-click range select
        self._search_after = None      # debounce handle for live search
        self._cols = 0
        self.current = None            # image in the inspector
        self.sugg = D.load_suggestions()
        self._edits: queue.Queue = queue.Queue()
        self._pending = 0
        self._relayout = False         # a queued edit removed/restored files: rebuild the page when drained
        threading.Thread(target=self._edit_worker, daemon=True).start()

        # ---- filter bar
        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 6))
        self.mode = ctk.CTkOptionMenu(bar, values=list(self.MODES), width=190,
                                      command=lambda _: self.goto(0, reset=True))
        self.mode.pack(side="left", padx=8, pady=8)
        self.defect = ctk.CTkOptionMenu(bar, values=["-"], width=150,
                                        command=lambda _: self.goto(0, reset=True))
        self.defect.pack(side="left", padx=(0, 8))
        self.search = ctk.CTkEntry(bar, placeholder_text="filename filter", width=130)
        self.search.pack(side="left", padx=(0, 8))
        self.search.bind("<Return>", lambda e: self.goto(0, reset=True))
        self.search.bind("<KeyRelease>", self.on_search_key)
        self.info = ctk.CTkLabel(bar, text="", text_color=DIM)
        self.info.pack(side="left", padx=8)
        ctk.CTkButton(bar, text="→", width=42, command=lambda: self.goto(self.page + 1)).pack(side="right", padx=(0, 8))
        ctk.CTkButton(bar, text="←", width=42, command=lambda: self.goto(self.page - 1)).pack(side="right", padx=4)
        self.btn_insp = ctk.CTkButton(bar, text="Inspector »", width=100, command=self.toggle_inspector)
        self.btn_insp.pack(side="right", padx=(4, 10))
        self.thumb = ctk.CTkSlider(bar, from_=96, to=220, number_of_steps=31, width=80,
                                   command=self.on_thumb)
        self.thumb.set(self.thumb_px())
        self.thumb.pack(side="right", padx=(0, 6))
        self.btn_ai = ctk.CTkButton(bar, text="AI pre-label", width=110, fg_color="transparent",
                                    border_width=1, border_color=INFO, text_color=INFO, command=self.ai_prelabel)
        self.btn_ai.pack(side="right", padx=4)

        # ---- dataset balance strip
        self.strip = ctk.CTkFrame(parent, fg_color=PANEL)
        self.strip.pack(fill="x", pady=(0, 6))

        # ---- bulk actions on the selection
        bulk = ctk.CTkFrame(parent, fg_color=PANEL)
        bulk.pack(fill="x", pady=(0, 6))
        self.selinfo = ctk.CTkLabel(bulk, text="0 selected", font=theme.H2)
        self.selinfo.pack(side="left", padx=10, pady=8)
        ctk.CTkButton(bulk, text="Select page", width=96, command=self.select_page).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Select all", width=86, command=self.select_all).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Clear", width=64, command=self.clear_sel).pack(side="left", padx=3)
        self.bdefect = ctk.CTkOptionMenu(bulk, values=["-"], width=170)
        self.bdefect.pack(side="left", padx=(14, 4))
        self.edit_btns = [
            ctk.CTkButton(bulk, text="Set defect", width=96, fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                          command=lambda: self.apply(1)),
            ctk.CTkButton(bulk, text="Clear defect", width=100, command=lambda: self.apply(0))]
        for b in self.edit_btns:
            b.pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Mark GOOD", width=96, fg_color="transparent", border_width=1, border_color=GOOD,
                      text_color=GOOD, command=self.mark_good).pack(side="left", padx=(10, 3))
        ctk.CTkButton(bulk, text="Accept AI", width=90, fg_color="transparent", border_width=1, border_color=INFO,
                      text_color=INFO, command=self.accept_ai).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Delete", width=70, fg_color="transparent", border_width=1, border_color=BAD,
                      text_color=BAD, command=self.delete).pack(side="left", padx=(10, 3))
        self.btn_undo = ctk.CTkButton(bulk, text="Undo", width=70, command=self.undo)
        self.btn_undo.pack(side="left", padx=3)

        ctk.CTkLabel(parent, text="1-9 toggle defect · G good · Enter accept AI · Del trash · Ctrl+Z undo · "
                                  "A page · Esc clear · ← → move · Shift-click range · double-click inspect",
                     text_color=MUTED, font=theme.TINY, anchor="w").pack(side="bottom", fill="x", padx=6, pady=(4, 0))

        # ---- grid + docked inspector
        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self.insp = ctk.CTkFrame(body, fg_color=PANEL, width=420)
        self.insp.pack_propagate(False)
        self.grid = ctk.CTkScrollableFrame(body, fg_color="transparent")
        self.grid.pack(side="left", fill="both", expand=True)
        self.grid.bind("<Configure>", self.on_resize)
        self._build_inspector()

        app.bind("<Key>", self.on_key)

    # settings -------------------------------------------------------------
    def thumb_px(self) -> int:
        return int(max(96, min(220, self.app.settings.get("thumb_px", 132))))

    def on_thumb(self, v):
        v = int(v)
        if v != self.app.settings.get("thumb_px"):
            self.app.settings["thumb_px"] = v
            D.save_settings(self.app.settings)
            if self._search_after is not None:
                self.app.after_cancel(self._search_after)
            self._search_after = self.app.after(300, self.load)

    def columns(self) -> int:
        fixed = int(self.app.settings.get("grid_columns", 0) or 0)
        if fixed > 0:
            return fixed
        w = self.grid.winfo_width()
        if w <= 1:
            w = 1100
        return max(1, (w - 20) // (self.thumb_px() + 18))

    def on_resize(self, _e=None):
        cols = self.columns()
        if cols != self._cols:
            self._cols = cols
            for i, p in enumerate(self.items):
                if p in self.cards:
                    self.cards[p]["card"].grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="nsew")

    # keyboard -------------------------------------------------------------
    def on_key(self, e):
        if self.app.tabs.get() != "Label":
            return
        try:
            if e.widget.winfo_toplevel() is not self.app:      # a dialog has focus
                return
            if e.widget.winfo_class() in ("Entry", "Text"):
                return
        except Exception:                                    # noqa: BLE001  (widget already destroyed)
            return
        k = e.keysym.lower()
        ctrl = bool(e.state & 0x0004)
        if ctrl and k == "z":
            self.undo()
        elif ctrl and k == "a":
            self.select_page()
        elif k in "123456789" and len(k) == 1:
            i = int(k) - 1
            if i < len(self.app.defects):
                self.toggle_defect(self.app.defects[i])
        elif k in ("g", "space"):
            self.mark_good()
        elif k in ("return", "kp_enter"):
            self.accept_ai()
        elif k == "delete":
            self.delete()
        elif k == "a":
            self.select_page()
        elif k == "escape":
            self.clear_sel()
        elif k == "left":
            self.step(-1) if self.inspecting else self.goto(self.page - 1)
        elif k == "right":
            self.step(1) if self.inspecting else self.goto(self.page + 1)

    def on_search_key(self, e):
        if self._search_after is not None:
            self.app.after_cancel(self._search_after)
        self._search_after = self.app.after(250, lambda: self.goto(0, reset=True))

    # data -----------------------------------------------------------------
    def refresh(self):
        ds = self.app.defects or ["-"]
        for m in (self.defect, self.bdefect):
            cur = m.get()
            m.configure(values=ds)
            m.set(cur if cur in ds else ds[0])
        for b in self.edit_btns:
            b.configure(state="normal" if self.app.defects else "disabled")
        self.sugg = D.load_suggestions()
        self.sel &= set(self.app.labels)
        self._build_checks()
        self.draw_strip()
        self.update_undo()
        self.load()

    def query(self):
        mode = self.MODES[self.mode.get()]
        d, q = self.defect.get(), self.search.get().strip().lower()
        items = self.sugg.get("items", {})
        out = []
        for p in sorted(self.app.labels, key=D._natkey):
            row = self.app.labels[p]
            if q and q not in p.lower():
                continue
            reviewed = row.get("reviewed", 1)
            if mode == "inbox" and reviewed:
                continue
            if mode == "suggested" and (reviewed or p not in items):
                continue
            if mode == "good" and not (reviewed and D.is_good(row, self.app.defects)):
                continue
            if mode == "defective" and D.is_good(row, self.app.defects):
                continue
            if mode in ("pos", "neg"):
                if d not in self.app.defects:
                    continue
                if bool(row.get(d, 0)) != (mode == "pos"):
                    continue
                if mode == "neg" and not reviewed:
                    continue
            out.append(p)
        return out

    def goto(self, page, reset=False):
        if reset:
            # A selection made under one filter must not ride along, unseen, into the next one.
            self.clear_sel()
        self.page = max(0, page)
        self.load()

    def load(self):
        all_paths = self.query()
        per = max(6, int(self.app.settings.get("per_page", PER_PAGE)))
        pages = max(1, -(-len(all_paths) // per))
        self.page = min(self.page, pages - 1)
        self.items = all_paths[self.page * per:(self.page + 1) * per]
        self.info.configure(text=f"{len(all_paths)}  ·  page {self.page + 1}/{pages}")

        for w in self.grid.winfo_children():
            w.destroy()
        self.cards.clear()
        self._imgs.clear()
        self._token += 1
        token = self._token

        px = self.thumb_px()
        self._cols = cols = self.columns()
        for i, p in enumerate(self.items):
            # Plain Tk widgets, not CustomTkinter: a page is ~60 cards, and a CTk widget draws itself onto a canvas
            # (about 4x the cost), which made paging and every label edit visibly slow.
            card = tk.Frame(self.grid, bg=PANEL, highlightthickness=2, highlightbackground=PANEL,
                            highlightcolor=PANEL)
            card.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="nsew")
            stripe = tk.Frame(card, height=4, bg=OFF)
            stripe.pack(fill="x", padx=3, pady=(3, 0))
            ph = tk.Label(card, bg=theme.VIDEO_BG, width=px, height=px, text="", bd=0, image=self._blank(px),
                          compound="center")
            ph.pack(padx=3, pady=(3, 0))
            badges = tk.Frame(card, bg=PANEL)
            badges.pack(fill="x", padx=3, pady=(3, 4))
            for w in (card, ph, stripe):
                w.bind("<Button-1>", lambda e, p=p: self.click(p, e))
                w.bind("<Double-Button-1>", lambda e, p=p: self.inspect(p))
            self.cards[p] = {"card": card, "ph": ph, "stripe": stripe, "badges": badges}
            self.paint(p)

        # Decoding 60 JPEGs blocks the window for a beat; do it off-thread and
        # drop the results if the user has already paged away. PIL work happens
        # there; the CTkImage (a Tk object) is built on the main thread.
        paths = list(self.items)

        def work():
            for p in paths:
                if token != self._token:
                    return
                b = D.thumbnail(p)
                if b is None:
                    continue
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                pil.thumbnail((px, px), Image.LANCZOS)
                self.app.post(lambda p=p, pil=pil: self.set_thumb(token, p, pil))
        threading.Thread(target=work, daemon=True).start()
        self.update_sel()
        if self.inspecting and self.current not in self.app.labels:
            self.show(self.items[0] if self.items else None)

    def set_thumb(self, token, path, pil):
        if token != self._token or path not in self.cards:
            return
        img = ImageTk.PhotoImage(pil)
        self._imgs.append(img)
        self.cards[path]["ph"].configure(image=img)

    # card look ------------------------------------------------------------
    def suggestion(self, p):
        """-> (defects the model would flag, top probability, confident?) or None."""
        probs = self.sugg.get("items", {}).get(p)
        if not probs:
            return None
        thr = self.app.cfg.get("thresholds", {})
        hits = [d for d, v in probs.items() if d in self.app.defects and v >= float(thr.get(d, 0.5))]
        top = max(probs.values()) if probs else 0.0
        conf = (all(v >= self.CONFIDENT or v <= 1 - self.CONFIDENT for v in probs.values())
                and len(hits) <= 1)
        return hits, top, conf

    def paint(self, p):
        c = self.cards.get(p)
        row = self.app.labels.get(p)
        if not c or row is None:
            return
        edge = ACC if p in self.sel else PANEL
        bg = theme.ACC_SOFT if p == self.current and self.inspecting else PANEL
        c["card"].configure(highlightbackground=edge, highlightcolor=edge, bg=bg)
        c["badges"].configure(bg=bg)
        sig = (tuple(self.app.labels[p].get(d, 0) for d in self.app.defects), row.get("reviewed", 1),
               str(self.suggestion(p)), bg)
        if c.get("sig") == sig:                       # selection changes repaint the border only, not the badges
            return
        c["sig"] = sig
        for w in c["badges"].winfo_children():
            w.destroy()
        tags = [d for d in self.app.defects if row.get(d)]
        reviewed = row.get("reviewed", 1)
        if not reviewed:
            colour = WARN
            self._badge(c["badges"], "NEW · not reviewed", WARN, theme.FAULT_SOFT)
        elif tags:
            colour = BAD
        else:
            colour = GOOD
            self._badge(c["badges"], "GOOD", GOOD, theme.PASS_SOFT)
        for d in tags:
            self._badge(c["badges"], f"{self.app.defects.index(d) + 1}  {d.replace('_', ' ')}", BAD,
                        theme.REJECT_SOFT)
        s = self.suggestion(p) if not reviewed else None
        if s:
            hits, top, conf = s
            txt = ("AI: " + ", ".join(h.replace("_", " ") for h in hits) if hits else "AI: GOOD") + f"  {top:.0%}"
            self._badge(c["badges"], txt + ("  ✓" if conf else ""), INFO, theme.INFO_SOFT)
            colour = INFO
        c["stripe"].configure(bg=colour)

    def _badge(self, parent, text, fg, bg):
        tk.Label(parent, text=text, fg=fg, bg=bg, font=theme.TINY, wraplength=self.thumb_px() - 10, justify="left",
                 anchor="w", padx=4, pady=1).pack(fill="x", pady=1)

    def _blank(self, px):
        """A transparent 1x1 image: it makes tk.Label's width/height mean pixels, not text characters."""
        if getattr(self, "_blank_img", None) is None:
            self._blank_img = ImageTk.PhotoImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        return self._blank_img

    # balance strip --------------------------------------------------------
    def draw_strip(self):
        for w in self.strip.winfo_children():
            w.destroy()
        c = self.app.counts or D.counts(self.app.defects, self.app.labels)
        thr = self.app.cfg.get("thresholds", {})

        r1 = ctk.CTkFrame(self.strip, fg_color="transparent")
        r1.pack(fill="x")
        r2 = ctk.CTkFrame(self.strip, fg_color="transparent")
        r2.pack(fill="x")

        def chip(text, n, colour, cmd, tip="", row=None):
            b = ctk.CTkButton(row or r1, text=f"{text} {n}", height=24, corner_radius=12, fg_color=PANEL_2,
                              hover_color=theme.ACC_SOFT, text_color=colour, font=theme.TINY,
                              border_width=1, border_color=colour if colour != INK else LINE, command=cmd,
                              width=30)
            b.pack(side="left", padx=2, pady=5)
            if tip:
                Tooltip(b, tip)

        ctk.CTkFrame(r1, width=6, height=1, fg_color="transparent").pack(side="left")
        chip("ALL", c.get("_total", 0), INK, lambda: self.filter_to("All images"))
        good = c.get("_good", 0)
        chip("GOOD", good, GOOD if good >= self.LOW_CLASS else WARN, lambda: self.filter_to("GOOD - passes"),
             "" if good >= self.LOW_CLASS else
             f"Only {good} good images. A real line is mostly good: the model has barely seen a pass.")
        chip("DEFECTIVE", c.get("_defective", 0), INK, lambda: self.filter_to("DEFECTIVE - any defect"))
        unrev = c.get("_unreviewed", 0)
        chip("INBOX", unrev, WARN if unrev else DIM, lambda: self.filter_to("Inbox - not reviewed"),
             "Not reviewed: excluded from training until someone labels them.")
        ctk.CTkFrame(r2, width=6, height=1, fg_color="transparent").pack(side="left")
        for i, d in enumerate(self.app.defects):
            n = c.get(d, 0)
            colour = BAD if n == 0 else WARN if n < self.LOW_CLASS else INK
            tip = (f"No images: this class is disabled at training (threshold "
                   f"{thr.get(d, 1.01)}) and can never fire." if n == 0 else
                   f"Only {n} images: too few to trust this class." if n < self.LOW_CLASS else "")
            chip(f"{i + 1}·{d.replace('_', ' ')}" if i < 9 else d.replace("_", " "), n, colour,
                 lambda d=d: self.filter_to("WITH this defect", d), tip, row=r2)

    def filter_to(self, mode, defect=None):
        self.mode.set(mode)
        if defect:
            self.defect.set(defect)
        self.goto(0, reset=True)

    # selection ------------------------------------------------------------
    def click(self, p, event):
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)
        if shift and self.last_click in self.items and p in self.items:
            lo, hi = sorted((self.items.index(self.last_click), self.items.index(p)))
            rng = self.items[lo:hi + 1]
            self.sel.update(rng)
            for q in rng:
                self.paint(q)
            self.update_sel()
        elif self.inspecting and not ctrl:
            self.show(p)                      # review mode: the inspected image IS the selection
        else:
            self.sel.symmetric_difference_update({p})
            self.paint(p)
            self.update_sel()
        self.last_click = p

    def select_page(self):
        self.sel.update(self.items)
        for p in self.items:
            self.paint(p)
        self.update_sel()

    def select_all(self):
        self.sel.update(self.query())
        for p in self.items:
            self.paint(p)
        self.update_sel()

    def clear_sel(self):
        was, self.sel = self.sel, set()
        for p in was:
            self.paint(p)
        self.update_sel()

    def offpage(self) -> int:
        return len(self.sel - set(self.items))

    def update_sel(self):
        off = self.offpage()
        self.selinfo.configure(text=f"{len(self.sel)} selected" + (f"  ({off} not on this page)" if off else ""),
                               text_color=WARN if off else INK)

    def need_sel(self) -> list | None:
        if not self.sel:
            messagebox.showinfo("Nothing selected", "Select some images first.")
            return None
        return sorted(self.sel, key=D._natkey)

    # edits ----------------------------------------------------------------
    def _edit_worker(self):
        while True:
            fn, paths = self._edits.get()
            try:
                fn()
                err = None
            except Exception as e:                       # noqa: BLE001
                err = e
            self.app.post(lambda paths=paths, err=err: self._edit_done(paths, err))

    def submit(self, fn, paths, local=None):
        """`local(row)` mirrors the edit in app.labels now; `fn` writes it to disk, in order, on the worker.
        local=None: the edit adds or removes files (delete, undo), so the page is rebuilt once it is written."""
        if local is None:
            self._relayout = True
        for p in paths:
            if local is not None and p in self.app.labels:
                local(self.app.labels[p])
                self.app.labels[p]["reviewed"] = 1
            self.paint(p)
        self._pending += 1
        self._edits.put((fn, list(paths)))
        if self.inspecting:
            self.show(self.current)

    def _edit_done(self, paths, err):
        self._pending -= 1
        if err is not None:
            messagebox.showerror("Label edit failed", f"{type(err).__name__}: {err}")
        if self._pending == 0:                           # re-read the truth once the queue has drained
            self.app.data_changed()
            self.sugg = D.load_suggestions()
            if self._relayout:
                self._relayout = False
                self.sel &= set(self.app.labels)
                self.load()
            else:
                for p in self.items:
                    self.paint(p)
            self.draw_strip()
            self.update_undo()
            if self.inspecting:
                self.show(self.current)

    def targets(self) -> list:
        return sorted(self.sel, key=D._natkey)

    def toggle_defect(self, d):
        paths = self.targets()
        if not paths:
            return
        on = not all(self.app.labels.get(p, {}).get(d) for p in paths)
        self._set_defect(paths, d, on)

    def _set_defect(self, paths, d, on):
        v = int(on)
        self.submit(lambda: D.apply_labels(paths, d, v), paths, lambda row: row.__setitem__(d, v))

    def apply(self, value):
        paths = self.need_sel()
        d = self.bdefect.get()
        if not paths or d not in self.app.defects:
            return
        if self.offpage() and not messagebox.askyesno(
                "Selection includes hidden images",
                f"{'Set' if value else 'Clear'} {d!r} on {len(paths)} images, {self.offpage()} of them NOT on "
                f"this page?"):
            return
        self._set_defect(paths, d, value)

    def mark_good(self):
        paths = self.targets()
        if not paths:
            return
        if self.offpage() and not messagebox.askyesno(
                "Selection includes hidden images",
                f"Mark {len(paths)} images GOOD, {self.offpage()} of them NOT on this page?"):
            return

        def local(row):
            for d in self.app.defects:
                row[d] = 0
        self.submit(lambda: (D.apply_labels(paths, clear_all=True), D.drop_suggestions(paths)), paths, local)
        self.after_label()

    def accept_ai(self):
        paths = [p for p in self.targets() if self.suggestion(p)]
        if not paths:
            return
        mapping = {p: self.suggestion(p)[0] for p in paths}

        def local_for(p):
            def local(row):
                for d in self.app.defects:
                    row[d] = int(d in mapping[p])
            return local
        for p in paths:                                  # mirror each one; one disk write for all
            if p in self.app.labels:
                local_for(p)(self.app.labels[p])
        self.submit(lambda: (D.set_labels(mapping, action="accept AI"), D.drop_suggestions(paths)),
                    paths, lambda row: None)
        self.after_label()

    def accept_confident(self):
        sure = [p for p in self.query() if not self.app.labels[p].get("reviewed", 1)
                and (self.suggestion(p) or (None, 0, False))[2]]
        if not sure:
            return messagebox.showinfo("Accept confident", "No suggestion in this view is confident enough "
                                                           f"(every probability ≥ {self.CONFIDENT:.0%} or "
                                                           f"≤ {1 - self.CONFIDENT:.0%}, at most one defect).")
        if not messagebox.askyesno("Accept confident suggestions",
                                   f"Accept the AI label for {len(sure)} images without looking at each?\n\n"
                                   f"They become training data. Spot-check a few first."):
            return
        self.sel = set(sure)
        self.accept_ai()

    def after_label(self):
        if self.inspecting and self.advance.get():
            self.step(1)

    def delete(self):
        paths = self.need_sel()
        if not paths:
            return
        off = self.offpage()
        if not messagebox.askyesno("Move to trash",
                                   f"Move {len(paths)} image(s) to the project trash?"
                                   + (f"\n\n{off} of them are NOT on this page." if off else "")
                                   + "\n\nUndo (Ctrl+Z) puts them back with their labels."):
            return
        self.sel.clear()
        self.update_sel()
        self.submit(lambda: D.delete_images(paths), paths)

    def update_undo(self):
        u = D.last_undoable()
        self.btn_undo.configure(state="normal" if u else "disabled",
                                text=f"Undo {u[1]} ({u[2]})" if u else "Undo")

    def undo(self):
        u = D.last_undoable()
        if not u or self._pending:          # undo only what is already on disk
            return
        self.submit(lambda: D.undo(u[0]), [])

    # inspector ------------------------------------------------------------
    @property
    def inspecting(self) -> bool:
        return self.insp.winfo_manager() != ""

    def toggle_inspector(self):
        if self.inspecting:
            self.insp.pack_forget()
            self.btn_insp.configure(text="Inspector »")
            cur, self.current = self.current, None
            if cur:
                self.paint(cur)
        else:
            self.insp.pack(side="right", fill="y", padx=(8, 0))
            self.btn_insp.configure(text="Inspector «")
            first = next(iter(sorted(self.sel & set(self.items), key=self.items.index)), None)
            self.show(first or (self.items[0] if self.items else None))

    def inspect(self, p):
        if not self.inspecting:
            self.toggle_inspector()
        self.show(p)

    def open_full(self, p):
        """Kept for other tabs (Train's misclassified images): show one image in the Label inspector."""
        if p not in self.app.labels:
            return
        self.app.tabs.set("Label")
        self.inspect(p)

    def _build_inspector(self):
        f = self.insp
        top = ctk.CTkFrame(f, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkLabel(top, text="INSPECTOR", font=theme.CAPS, text_color=DIM).pack(side="left")
        self.i_pos = ctk.CTkLabel(top, text="", font=theme.SMALL, text_color=DIM)
        self.i_pos.pack(side="right")
        self.i_img = ctk.CTkLabel(f, text="no image", width=400, height=260, fg_color=theme.VIDEO_BG,
                                  corner_radius=6, text_color=MUTED)
        self.i_img.pack(padx=10)
        self.i_name = ctk.CTkLabel(f, text="", font=theme.SMALL, text_color=DIM, wraplength=390, justify="left")
        self.i_name.pack(anchor="w", padx=10, pady=(4, 0))
        self.i_state = ctk.CTkLabel(f, text="", font=theme.H2)
        self.i_state.pack(anchor="w", padx=10, pady=(2, 4))
        nav = ctk.CTkFrame(f, fg_color="transparent")
        nav.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkButton(nav, text="« Prev", width=80, command=lambda: self.step(-1)).pack(side="left")
        ctk.CTkButton(nav, text="GOOD  (G)", width=110, fg_color="transparent", border_width=1, border_color=GOOD,
                      text_color=GOOD, command=self.mark_good).pack(side="left", padx=6)
        ctk.CTkButton(nav, text="Next »", width=80, command=lambda: self.step(1)).pack(side="left")
        self.advance = ctk.CTkCheckBox(nav, text="auto-advance", font=theme.SMALL)
        self.advance.pack(side="right")
        self.advance.select()
        # pinned to the bottom first, so the defect list (which grows with the classes) gets what is left
        ctk.CTkButton(f, text="Accept all confident AI labels in this view…", fg_color="transparent",
                      border_width=1, border_color=INFO, text_color=INFO,
                      command=self.accept_confident).pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.i_queue = ctk.CTkLabel(f, text="", font=theme.SMALL, text_color=DIM)
        self.i_queue.pack(side="bottom", anchor="w", padx=10, pady=(0, 4))
        ai = ctk.CTkFrame(f, fg_color=theme.INFO_SOFT, corner_radius=6)
        ai.pack(side="bottom", fill="x", padx=10, pady=(4, 6))
        self.i_ai = ctk.CTkLabel(ai, text="no AI suggestion", text_color=INFO, font=theme.SMALL, wraplength=250,
                                 justify="left")
        self.i_ai.pack(side="left", padx=8, pady=6)
        self.i_ai_btn = ctk.CTkButton(ai, text="Accept (Enter)", width=110, fg_color=INFO, text_color=theme.BG,
                                      hover_color=INFO, command=self.accept_ai)
        self.i_ai_btn.pack(side="right", padx=8, pady=6)
        ctk.CTkLabel(f, text="DEFECTS   (click or press the number)", font=theme.CAPS, text_color=DIM).pack(
            anchor="w", padx=10, pady=(4, 0))
        self.i_checks_box = ctk.CTkScrollableFrame(f, fg_color="transparent")
        self.i_checks_box.pack(fill="both", expand=True, padx=6)
        self.i_checks: dict = {}

    def _build_checks(self):
        for w in self.i_checks_box.winfo_children():
            w.destroy()
        self.i_checks = {}
        for i, d in enumerate(self.app.defects):
            cb = ctk.CTkCheckBox(self.i_checks_box, text=f"{i + 1}  {d.replace('_', ' ')}" if i < 9
                                 else f"     {d.replace('_', ' ')}",
                                 fg_color=BAD, hover_color=BAD, font=theme.SMALL, checkbox_width=20,
                                 checkbox_height=20, width=190,
                                 command=lambda d=d: self.check_clicked(d))
            cb.grid(row=i // 2, column=i % 2, sticky="w", pady=2, padx=4)
            self.i_checks[d] = cb

    def check_clicked(self, d):
        if self.current is None:
            return
        on = bool(self.i_checks[d].get())
        self._set_defect([self.current], d, on)

    def step(self, delta):
        if not self.items:
            return
        if self.current in self.items:
            i = self.items.index(self.current) + delta
        else:
            i = 0
        if i >= len(self.items):
            if self.page * max(6, int(self.app.settings.get("per_page", PER_PAGE))) + len(self.items) \
                    < len(self.query()):
                self.goto(self.page + 1)
                self.show(self.items[0] if self.items else None)
            return
        if i < 0:
            if self.page > 0:
                self.goto(self.page - 1)
                self.show(self.items[-1] if self.items else None)
            return
        self.show(self.items[i])

    def show(self, p):
        old, self.current = self.current, p
        if p is not None:
            was = self.sel
            self.sel = {p}
            for q in was | {p}:
                self.paint(q)
            self.update_sel()
        if old and old != p:
            self.paint(old)
        if p is None or p not in self.app.labels:
            self.i_img.configure(image=None, text="no image")
            self.i_name.configure(text="")
            self.i_state.configure(text="")
            return
        img = D.imread(D.IMAGE_ROOT / p)
        if img is not None:
            h, w = img.shape[:2]
            s = min(400 / w, 260 / h)
            im = bgr_to_ctk(cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                                       interpolation=cv2.INTER_AREA))
            self._insp_img = im
            self.i_img.configure(image=im, text="")
        else:
            self.i_img.configure(image=None, text="cannot read image")
        row = self.app.labels[p]
        tags = [d for d in self.app.defects if row.get(d)]
        reviewed = row.get("reviewed", 1)
        self.i_name.configure(text=f"{Path(p).name}\n{Path(p).parent.as_posix()}/")
        self.i_state.configure(text=("NEW · not reviewed" if not reviewed else
                                     ("REJECT: " + ", ".join(tags)) if tags else "GOOD"),
                               text_color=WARN if not reviewed else BAD if tags else GOOD)
        for d, cb in self.i_checks.items():
            cb.select() if row.get(d) else cb.deselect()
        s = self.suggestion(p)
        if s and not reviewed:
            hits, top, conf = s
            probs = self.sugg["items"][p]
            lines = ", ".join(f"{d.replace('_', ' ')} {probs[d]:.0%}" for d in hits) if hits else "GOOD"
            self.i_ai.configure(text=f"AI ({self.sugg.get('model') or '?'}): {lines}"
                                     + ("\nconfident" if conf else "\nNOT confident: check it"))
            self.i_ai_btn.configure(state="normal")
        else:
            self.i_ai.configure(text="no AI suggestion" if reviewed else "no AI suggestion: run AI pre-label")
            self.i_ai_btn.configure(state="disabled")
        n_new = sum(1 for q in self.items if not self.app.labels.get(q, {}).get("reviewed", 1))
        pos = self.items.index(p) + 1 if p in self.items else 0
        self.i_pos.configure(text=f"{pos} / {len(self.items)} on page")
        self.i_queue.configure(text=f"{n_new} not reviewed on this page   ·   "
                                    f"{self.app.counts.get('_unreviewed', 0)} in the inbox")

    # AI pre-label ---------------------------------------------------------
    def ai_prelabel(self):
        stamp = self.app.cfg.get("active_model")
        if not stamp:
            return messagebox.showinfo("AI pre-label", "This product has no trained model yet. Label a first "
                                                       "batch by hand, train, then let the model pre-label the rest.")
        inbox = [p for p, r in self.app.labels.items() if not r.get("reviewed", 1)]
        if not inbox:
            return messagebox.showinfo("AI pre-label", "The inbox is empty: nothing to pre-label.")
        btn = self.btn_ai

        def work():
            m = infer.Model(stamp)
            missing = [d for d in self.app.defects if d not in m.defects]
            out = {}
            for i, p in enumerate(inbox):
                img = D.imread(D.IMAGE_ROOT / p)
                if img is not None:
                    probs = m.predict(img)
                    out[p] = {d: round(v, 4) for d, v in probs.items() if d in self.app.defects}
                if i % 10 == 0:
                    self.app.post(lambda i=i: btn.configure(text=f"AI… {i}/{len(inbox)}"))
            d = D.load_suggestions()
            d.update(model=stamp, time=time.strftime("%Y-%m-%d %H:%M:%S"))
            d["items"].update(out)
            D.save_suggestions(d)
            return len(out), missing

        def done(r):
            n, missing = r
            btn.configure(text="AI pre-label", state="normal")
            self.sugg = D.load_suggestions()
            self.filter_to("AI suggested - not reviewed")
            msg = (f"{n} inbox images pre-labelled by model {stamp}.\n\nThey stay NOT reviewed (and out of "
                   f"training) until you accept or correct each one: Enter accepts, 1-9 corrects.")
            if missing:
                msg += (f"\n\nThe model was trained before these classes existed and cannot suggest them: "
                        f"{', '.join(missing)}. Retrain to include them.")
            messagebox.showinfo("AI pre-label", msg)

        btn.configure(state="disabled", text="AI… loading model")
        self.app.run_bg(work, done)


class Tooltip:
    """Hover text for a widget (Tk has none built in)."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _e=None):
        if self.tip or not self.text:
            return
        x, y = self.widget.winfo_rootx() + 10, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, bg=PANEL_2, fg=INK, font=theme.SMALL, wraplength=320, justify="left",
                 padx=8, pady=4, bd=1, relief="solid").pack()

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ------------------------------------------------------------------ Defects
class DefectsTab:
    def __init__(self, app: App, parent):
        self.app = app
        left = ctk.CTkFrame(parent, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.list.pack(fill="both", expand=True)

        right = ctk.CTkFrame(parent, fg_color=PANEL, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="ADD A DEFECT TYPE", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
        ctk.CTkLabel(right, wraplength=300, justify="left", text_color=DIM,
                     text="A defect is a column in labels.csv. Adding one sets it to 0 for "
                          "every existing image; then use the -ve filter to find and flag "
                          "the positives.").pack(anchor="w", padx=14)
        self.entry = ctk.CTkEntry(right, placeholder_text="e.g. loose cap")
        self.entry.pack(fill="x", padx=14, pady=(12, 6))
        self.entry.bind("<Return>", lambda e: self.add())
        ctk.CTkButton(right, text="Add", fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.add).pack(fill="x", padx=14)

    def refresh(self):
        for w in self.list.winfo_children():
            w.destroy()
        c = self.app.counts
        good, bad = c.get("_good", 0), c.get("_defective", 0)
        unrev = c.get("_unreviewed", 0)

        # The dataset-level split -- the "+ve / -ve dataset" in the ordinary
        # sense: a bottle either passes or it does not. Shown on its own,
        # above and apart from the per-defect table, because the two mean
        # different things and sharing a word for them caused real confusion.
        top = ctk.CTkFrame(self.list, fg_color=PANEL)
        top.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(top, text="DATASET SPLIT", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 4))
        for title, n, col in (("GOOD  (+ve, passes)", good, GOOD),
                              ("DEFECTIVE  (-ve, fails)", bad, BAD),
                              ("NOT YET REVIEWED", unrev, WARN if unrev else DIM)):
            box = ctk.CTkFrame(row, fg_color="transparent")
            box.pack(side="left", padx=(0, 40))
            ctk.CTkLabel(box, text=str(n), text_color=col,
                         font=("Segoe UI", 26, "bold")).pack(anchor="w")
            ctk.CTkLabel(box, text=title, text_color=DIM,
                         font=("Segoe UI", 13)).pack(anchor="w")
            if title.startswith("GOOD"):
                ctk.CTkButton(box, text="Upload GOOD images", width=170, fg_color=ACC,
                              text_color=ACC_T, hover_color=ACC_H,
                              command=lambda: self.upload(None)).pack(anchor="w", pady=(6, 0))
        share = good / max(1, good + bad)
        ctk.CTkLabel(top, wraplength=700, justify="left", padx=0,
                     text_color=WARN if share < 0.5 else GOOD,
                     text=(f"Only {share:.0%} of the labelled set passes. On a real line good is "
                           f"the overwhelming majority, so the model is seeing a world where "
                           f"defects are normal — grow the good set."
                           if share < 0.5 else
                           f"{share:.0%} of the labelled set passes.")
                     ).pack(anchor="w", padx=14, pady=(0, 12))

        ctk.CTkLabel(self.list, text="PER-DEFECT", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self.list, text_color=DIM, justify="left", wraplength=700,
                     text="How many images carry each defect. A bottle can have several, so these "
                          "add up to more than the defective count above.\n"
                          "WITHOUT is not the same as good: it is every other image, including "
                          f"the {c.get('_defective', 0)} that fail on a different defect."
                     ).pack(anchor="w", pady=(0, 6))

        hdr = ctk.CTkFrame(self.list, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 4))
        for t, wdt in (("DEFECT", 220), ("WITH", 80), ("WITHOUT", 90), ("STATUS", 330)):
            ctk.CTkLabel(hdr, text=t, width=wdt, anchor="w", text_color=DIM,
                         font=("Segoe UI", 13, "bold")).pack(side="left")

        for i, d in enumerate(self.app.defects):
            n = c.get(d, 0)
            row = ctk.CTkFrame(self.list, fg_color=PANEL)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f" {i + 1}   {d}", width=220, anchor="w").pack(side="left", pady=6)
            ctk.CTkLabel(row, text=str(n), width=80, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=str(c["_total"] - n), width=90, anchor="w").pack(side="left")
            if n == 0:
                msg, col = "no images - disabled, cannot be learned", BAD
            elif n < 40:
                msg, col = "thin - aim for 100+", WARN
            else:
                msg, col = "ok", GOOD
            ctk.CTkLabel(row, text=msg, width=330, anchor="w", text_color=col).pack(side="left")
            ctk.CTkButton(row, text="Delete", width=70, fg_color="transparent", border_width=1,
                          text_color=BAD, command=lambda d=d: self.delete(d)).pack(side="right", padx=(4, 8))
            ctk.CTkButton(row, text="Rename", width=76, fg_color="transparent", border_width=1, text_color=ACC,
                          command=lambda d=d: self.rename(d)).pack(side="right", padx=4)
            ctk.CTkButton(row, text="Upload", width=76, fg_color="transparent", border_width=1,
                          text_color=ACC, command=lambda d=d: self.upload(d)).pack(side="right", padx=4)

    def upload(self, defect: str | None):
        """Copy images in. defect None means the +ve (good) folder."""
        where = f"{D.NEG_DIR}/{defect}" if defect else D.POS_DIR
        what = f"'{defect}'" if defect else "GOOD (no defect)"
        folder = filedialog.askdirectory(title=f"Folder of images for {what} — Cancel to pick files")
        srcs = [folder] if folder else None
        if not srcs:
            files = filedialog.askopenfilenames(
                title=f"Images for {what}",
                filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
            if not files:
                return
            srcs = list(files)

        def work():
            if len(srcs) == 1 and Path(srcs[0]).is_dir():
                return D.add_folder(srcs[0], where)
            return D.add_images(srcs, where)

        def done(n):
            self.app.reload()
            messagebox.showinfo("Upload", f"{n} image(s) copied into {where}/."
                                if n else "No images found there.")
        self.app.run_bg(work, done)

    def add(self):
        name = self.entry.get().strip()
        if not name:
            return
        try:
            D.add_defect(name)
        except Exception as e:
            return messagebox.showerror("Cannot add", str(e))
        self.entry.delete(0, "end")
        self.app.reload()

    def rename(self, d):
        dlg = ctk.CTkInputDialog(text=f"New name for {d}", title="Rename defect")
        new = dlg.get_input()
        if not new:
            return
        try:
            D.rename_defect(d, new)
        except Exception as e:
            return messagebox.showerror("Cannot rename", str(e))
        self.app.reload()

    def delete(self, d):
        n = self.app.counts.get(d, 0)
        if not messagebox.askyesno(
                "Delete defect type",
                f"Delete the '{d}' column and its {n} label(s)?\n\n"
                f"The images are kept. Any sitting in -ve/{d}/ move to the inbox as "
                f"unreviewed, so nothing silently becomes a good bottle.\n\nThe labels are gone."):
            return
        D.delete_defect(d)
        self.app.reload()


# -------------------------------------------------------------------- Train
#: Model Selection choices -- friendly label -> train.BACKBONES key.
#: EfficientNet-B0 stays the default: it's what this project has always
#: trained on, and every other option here is for benchmarking against it,
#: not replacing it.
BACKBONE_LABELS = {
    "EfficientNet-B0 (default)": "efficientnet_b0",
    "EfficientNet-B1": "efficientnet_b1",
    "MobileNetV3-Small": "mobilenet_v3_small",
    "ResNet18": "resnet18",
    "ConvNeXt-Tiny": "convnext_tiny",
}


class TrainTab:
    def __init__(self, app: App, parent):
        self.app, self.busy = app, False
        self._live_hist: list[dict] = []   # this session's epochs, for the live charts
        self._prev_metrics: dict | None = None  # active model's metrics before this run started

        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(bar, text="Epochs").pack(side="left", padx=(10, 6), pady=8)
        self.epochs = ctk.CTkEntry(bar, width=60)
        self.epochs.insert(0, "25")
        self.epochs.pack(side="left")
        ctk.CTkLabel(bar, text="Batch").pack(side="left", padx=(10, 6))
        self.batch = ctk.CTkEntry(bar, width=60)
        self.batch.insert(0, "32")
        self.batch.pack(side="left")
        ctk.CTkLabel(bar, text="LR").pack(side="left", padx=(10, 6))
        self.lr = ctk.CTkEntry(bar, width=70)
        self.lr.insert(0, "3e-4")
        self.lr.pack(side="left")
        self.btn = ctk.CTkButton(bar, text="Train now", fg_color=ACC, text_color=ACC_T,
                                 hover_color=ACC_H, command=self.start)
        self.btn.pack(side="left", padx=10)
        ctk.CTkLabel(bar, text_color=DIM,
                     text="Unreviewed images are excluded. The split is by detected scene, never random.",
                     ).pack(side="left", padx=6)

        sel = ctk.CTkFrame(parent, fg_color=PANEL)
        sel.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(sel, text="MODEL SELECTION", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(side="left", padx=(10, 10), pady=8)
        self.arch = ctk.CTkOptionMenu(sel, values=list(BACKBONE_LABELS), width=210)
        self.arch.pack(side="left")
        ctk.CTkLabel(sel, text_color=DIM,
                     text="Compares against EfficientNet-B0, the current production model - "
                          "training a different backbone never replaces it, only adds another "
                          "checkpoint to benchmark.").pack(side="left", padx=10)
        self.imgsize = ctk.CTkLabel(sel, text_color=DIM)
        self.imgsize.pack(side="right", padx=10)

        self.summary = ctk.CTkLabel(parent, text="No training run yet.", text_color=DIM,
                                    justify="left", anchor="w")
        self.summary.pack(fill="x", pady=(0, 8))

        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        curves = ctk.CTkFrame(left, fg_color="transparent")
        curves.pack(fill="x", pady=(0, 8))
        self.c_loss = chart_panel(curves, "TRAINING LOSS PER EPOCH", 470, 190,
                                  "Fills in live while training runs.")
        self.c_f1 = chart_panel(curves, "VALIDATION MACRO-F1 PER EPOCH", 470, 190,
                                "The checkpoint kept is the best epoch, not the last one.")

        self.log = ctk.CTkTextbox(left, height=180, font=MONO, fg_color=VIDEO_BG,
                                   text_color=INK)
        self.log.pack(fill="x")
        self.log.insert("end", "No training run in this session.\n")
        self.metrics = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.metrics.pack(fill="both", expand=True, pady=(8, 0))

        right = ctk.CTkFrame(body, fg_color=PANEL, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="MODEL VERSIONS", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
        self.models = ctk.CTkScrollableFrame(right, fg_color="transparent", height=220)
        self.models.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkLabel(right, text="BENCHMARK", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(0, 2))
        ctk.CTkLabel(right, text_color=DIM, font=("Segoe UI", 12), wraplength=310, justify="left",
                     text="Each model's own stored numbers, one row per checkpoint - lets "
                          "different backbones be compared side by side.").pack(anchor="w", padx=14, pady=(0, 6))
        self.bench = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.bench.pack(fill="both", expand=True, padx=8, pady=(0, 10))

    def refresh(self):
        cfg = self.app.cfg
        iw, ih = D.input_wh(cfg)
        self.imgsize.configure(text=f"Image size: {iw}x{ih} (fixed to the calibrated ROI)")

        for w in self.models.winfo_children():
            w.destroy()
        active = cfg.get("active_model")
        stamps = infer.list_models()
        if not stamps:
            ctk.CTkLabel(self.models, text="None yet.", text_color=DIM).pack(anchor="w")
        metas = {}
        for s in stamps:
            mp = D.MODELS / s / "metrics.json"
            metas[s] = json.loads(mp.read_text()) if mp.exists() else {}
            row = ctk.CTkFrame(self.models, fg_color="transparent")
            row.pack(fill="x", pady=2)
            label = f"{s}  ·  {metas[s].get('arch', '?')}"
            ctk.CTkLabel(row, text=label, text_color=ACC if s == active else None,
                         font=("Segoe UI", 14, "bold" if s == active else "normal")).pack(side="left")
            if s == active:
                ctk.CTkLabel(row, text="active", text_color=DIM, font=("Segoe UI", 12)).pack(side="left", padx=6)
            else:
                ctk.CTkButton(row, text="Use", width=48, height=24,
                              command=lambda s=s: self.activate(s)).pack(side="right")

        for w in self.bench.winfo_children():
            w.destroy()
        if stamps:
            hdr = ctk.CTkFrame(self.bench, fg_color="transparent")
            hdr.pack(fill="x")
            for t in ("MODEL", "ACC", "PREC", "REC", "F1", "TRAIN s", "INFER ms", "FPS", "SIZE MB"):
                ctk.CTkLabel(hdr, text=t, width=68, anchor="w", text_color=DIM,
                             font=("Segoe UI", 12, "bold")).pack(side="left")
            for s in stamps:
                m = metas[s]
                seconds = sum(hh.get("seconds", 0) for hh in m.get("history", []))
                row = ctk.CTkFrame(self.bench, fg_color=PANEL)
                row.pack(fill="x", pady=1)
                vals = (m.get("arch", "?")[:10], m.get("accuracy", "-"), m.get("macro_precision", "-"),
                        m.get("macro_recall", "-"), m.get("macro_f1", "-"),
                        round(seconds) if seconds else "-", m.get("infer_ms", "-"),
                        m.get("fps", "-"), m.get("model_size_mb", "-"))
                for v in vals:
                    ctk.CTkLabel(row, text=str(v), width=68, anchor="w").pack(side="left", pady=4)

        self.show_metrics()

    def show_metrics(self):
        for w in self.metrics.winfo_children():
            w.destroy()
        active = self.app.cfg.get("active_model")
        if not active:
            self.summary.configure(text="No training run yet.")
            self.c_loss.delete("all")
            self.c_f1.delete("all")
            return
        p = D.MODELS / active / "metrics.json"
        if not p.exists():
            return
        import json
        m = json.loads(p.read_text())

        hist = m.get("history") or []
        if hist:
            xs = [h["epoch"] for h in hist]
            charts.line_chart(self.c_loss, {"loss": [h["loss"] for h in hist]},
                              x_values=xs, legend=False)
            charts.line_chart(self.c_f1, {"macro-F1": [h["macro_f1"] for h in hist]},
                              x_values=xs, y_lo=0, y_hi=1, legend=False)

        seconds = sum(h.get("seconds", 0) for h in hist)
        n_scored = sum(1 for d in m["defects"] if m["per_defect"][d]["n_pos"])
        text = (f"{m['epochs']} epochs   ·   {seconds:.0f}s total   ·   macro-F1 {m['macro_f1']}"
                f"   ·   train {m['n_train']} / val {m['n_val']} images"
                f"   ·   {n_scored}/{len(m['defects'])} defects scored")
        prev = self._prev_metrics
        if prev:
            deltas = []
            for d in m["defects"]:
                cur, old = m["per_defect"].get(d, {}), prev.get("per_defect", {}).get(d, {})
                if cur.get("n_pos") and old.get("n_pos"):
                    deltas.append((d, cur["recall"] - old["recall"]))
            if deltas:
                gain = max(deltas, key=lambda x: x[1])
                drop = min(deltas, key=lambda x: x[1])
                if gain[1] > 0:
                    text += f"   ·   biggest gain: {gain[0]} +{gain[1]:.2f} recall"
                if drop[1] < 0:
                    text += f"   ·   biggest drop: {drop[0]} {drop[1]:.2f} recall"
        self.summary.configure(text=text)

        scored = [d for d in m["defects"] if m["per_defect"][d]["n_pos"]]
        if scored:
            pickbar = ctk.CTkFrame(self.metrics, fg_color="transparent")
            pickbar.pack(fill="x", pady=(0, 6))
            ctk.CTkLabel(pickbar, text="Confusion for:").pack(side="left", padx=(0, 6))
            pick = ctk.CTkOptionMenu(pickbar, values=scored, width=170)
            pick.pack(side="left")
            pick.set(scored[0])

            row = ctk.CTkFrame(self.metrics, fg_color="transparent")
            row.pack(fill="x", pady=(0, 8))
            c_bars = chart_panel(row, "RECALL BY DEFECT", 470, 210,
                                 "Recall is the number that matters - a missed defect ships.")
            charts.bar_chart(c_bars, [d[:12] for d in scored],
                             [m["per_defect"][d]["recall"] for d in scored],
                             colours=[charts.GREEN if m["per_defect"][d]["recall"] >= 0.9
                                      else charts.AMBER if m["per_defect"][d]["recall"] >= 0.7
                                      else charts.RED for d in scored],
                             y_hi=1.0)
            c_conf = chart_panel(row, "CONFUSION MATRIX (selected defect)", 470, 210,
                                 "One 2x2 per defect: every defect is its own yes/no question "
                                 "about the same bottle.")

            def draw_conf(d):
                x = m["per_defect"][d]
                tp, fp, fn = x.get("tp", 0), x.get("fp", 0), x.get("fn", 0)
                tn = max(0, m.get("n_val", tp + fp + fn) - tp - fp - fn)
                charts.confusion(c_conf, tp, fp, fn, tn)

            pick.configure(command=draw_conf)
            draw_conf(scored[0])

        hdr = ctk.CTkFrame(self.metrics, fg_color="transparent")
        hdr.pack(fill="x")
        for t, w in (("DEFECT", 170), ("VAL+", 70), ("THR", 60), ("PREC", 80),
                     ("RECALL", 80), ("F1", 80), ("MISSED", 80)):
            ctk.CTkLabel(hdr, text=t, width=w, anchor="w", text_color=DIM,
                         font=("Segoe UI", 13, "bold")).pack(side="left")
        for d in m["defects"]:
            x = m["per_defect"][d]
            row = ctk.CTkFrame(self.metrics, fg_color=PANEL)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=" " + d, width=170, anchor="w").pack(side="left", pady=5)
            if not x["n_pos"]:
                ctk.CTkLabel(row, text=x.get("note", ""), anchor="w",
                             text_color=BAD if not x.get("n_train_pos") else WARN).pack(side="left")
                continue
            r = x["recall"]
            col = GOOD if r >= 0.9 else WARN if r >= 0.7 else BAD
            for txt, w, c in ((x["n_pos"], 70, None), (x["threshold"], 60, None),
                              (f"{x['precision']:.3f}", 80, None), (f"{r:.3f}", 80, col),
                              (f"{x['f1']:.3f}", 80, None),
                              (x.get("fn", 0), 80, BAD if x.get("fn") else None)):
                ctk.CTkLabel(row, text=str(txt), width=w, anchor="w", text_color=c).pack(side="left")

        ctk.CTkLabel(self.metrics, text_color=DIM, justify="left", wraplength=700,
                     text="Recall is the number that matters - a missed defect ships. MISSED is "
                          "false negatives on validation.").pack(anchor="w", pady=(6, 10))

        bad = m.get("mistakes", [])
        if bad:
            ctk.CTkLabel(self.metrics, text=f"SHOW ME THE MISTAKES  ({len(bad)})",
                         text_color=DIM, font=("Segoe UI", 13, "bold")).pack(anchor="w")
            ctk.CTkLabel(self.metrics, text_color=DIM, justify="left", wraplength=700,
                         text="Sorted by how confident the model was while being wrong. Expect a "
                              "good share to be mislabelled rather than mispredicted - fix the "
                              "label and retrain. Double-click to open.").pack(anchor="w", pady=(0, 6))
            grid = ctk.CTkFrame(self.metrics, fg_color="transparent")
            grid.pack(fill="x")
            self._mimgs = []
            for i, x in enumerate(bad[:27]):
                card = ctk.CTkFrame(grid, fg_color=PANEL)
                card.grid(row=i // 9, column=i % 9, padx=4, pady=4)
                lab = ctk.CTkLabel(card, text="", width=110, height=110)
                lab.pack(padx=4, pady=(4, 0))
                b = D.thumbnail(x["path"])
                if b is not None:
                    arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        im = bgr_to_ctk(arr, (110, 110))
                        self._mimgs.append(im)
                        lab.configure(image=im)
                kind = "missed" if x["kind"] == "false_neg" else "false alarm"
                ctk.CTkLabel(card, text=f"{kind}\n{x['defect']} {x['prob']}",
                             font=("Segoe UI", 12),
                             text_color=BAD if x["kind"] == "false_neg" else WARN).pack(pady=(2, 5))
                for w in (card, lab):
                    w.bind("<Double-Button-1>",
                           lambda e, p=x["path"]: self.app.tab_label.open_full(p))

    def activate(self, stamp):
        import json
        cfg = D.load_config()
        cfg["active_model"] = stamp
        mp = D.MODELS / stamp / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            cfg["thresholds"] = {d: v["threshold"] for d, v in m["per_defect"].items()}
        D.save_config(cfg)
        self._prev_metrics = None      # switching checkpoints isn't "this training run vs last"
        try:
            self.app.cams.load_model(stamp)
        except Exception:
            pass
        self.app.reload()

    def start(self):
        if self.busy:
            return
        try:
            n = int(self.epochs.get())
            b = int(self.batch.get())
            lr = float(self.lr.get())
        except ValueError:
            return messagebox.showerror("Training parameters",
                                        "Epochs/Batch must be whole numbers and LR a number.")
        arch = BACKBONE_LABELS[self.arch.get()]
        self.busy = True
        self.btn.configure(state="disabled", text="Training…")
        self.log.delete("1.0", "end")
        self.summary.configure(text=f"Training {n} epoch(s) on {arch}…")

        # captured before train.run() overwrites active_model, so the summary
        # line can report what changed relative to the model this replaces
        prev_stamp = self.app.cfg.get("active_model")
        prev_path = D.MODELS / prev_stamp / "metrics.json" if prev_stamp else None
        try:
            self._prev_metrics = json.loads(prev_path.read_text()) if prev_path and prev_path.exists() else None
        except Exception:
            self._prev_metrics = None
        self._live_hist = []

        def emit(line):
            self.app.post(lambda: (self.log.insert("end", str(line) + "\n"), self.log.see("end")))

        def on_epoch(rec):
            self.app.post(lambda: self._draw_live(rec))

        def work():
            import train
            try:
                train.run(epochs=n, batch=b, lr=lr, arch=arch, log=emit, on_epoch=on_epoch)
            except Exception as e:
                emit(f"ERROR: {type(e).__name__}: {e}")
            finally:
                self.app.post(self.finish)
        threading.Thread(target=work, daemon=True).start()

    def _draw_live(self, rec):
        self._live_hist.append(rec)
        xs = [h["epoch"] for h in self._live_hist]
        charts.line_chart(self.c_loss, {"loss": [h["loss"] for h in self._live_hist]},
                          x_values=xs, legend=False)
        charts.line_chart(self.c_f1, {"macro-F1": [h["macro_f1"] for h in self._live_hist]},
                          x_values=xs, y_lo=0, y_hi=1, legend=False)

    def finish(self):
        self.busy = False
        self.btn.configure(state="normal", text="Train now")
        self.app.reload()


# --------------------------------------------------------------------- Live
class LiveTab:
    """Several cameras at once, each scored independently, plus one verdict.

    The combined verdict rejects if ANY camera rejects -- the cameras are angles
    on one bottle, and a defect only the side camera can see is still a defect.
    """

    MODES = ("Classifier", "Classifier + YOLO", "YOLO only")

    def __init__(self, app: App, parent):
        self.app = app
        self._detector = None                 # loaded once, shared by every camera
        self.sources: list[tuple[str, object]] = []
        self.picked: dict[str, ctk.CTkCheckBox] = {}
        self.panes: dict[str, ctk.CTkLabel] = {}
        self.running = False
        self._imgs: dict[str, object] = {}
        self.checks: dict[str, ctk.CTkCheckBox] = {}
        self.sliders: dict[str, ctk.CTkSlider] = {}

        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        self.btn_start = ctk.CTkButton(bar, text="Start", width=80, fg_color=ACC,
                                       text_color=ACC_T, hover_color=ACC_H,
                                       command=self.start)
        self.btn_start.pack(side="left", padx=(10, 4), pady=8)
        ctk.CTkButton(bar, text="Stop", width=70, command=self.stop).pack(side="left")
        ctk.CTkButton(bar, text="⟳ Scan", width=80, command=self.scan).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Open a video…", command=self.pick_video).pack(side="left", padx=6)
        # What runs on each frame. The YOLO detector finds bottle / cap / label boxes only: it
        # does not decide defects, so "YOLO only" is a test mode that reports FAULT by design.
        self.infer_mode = ctk.CTkOptionMenu(bar, values=list(self.MODES), width=170)
        self.infer_mode.set(self.MODES[1])      # classifier + detector: the detector says WHEN a bottle is in view
        self.infer_mode.pack(side="left", padx=6)
        self.verdict = ctk.CTkLabel(bar, text="", font=("Segoe UI", 22, "bold"))
        self.verdict.pack(side="left", padx=20)
        self.detlabel = ctk.CTkLabel(bar, text="", text_color=DIM)
        self.detlabel.pack(side="left", padx=4)
        # Operators see ONE verdict per bottle. Boxes, score bars and detector numbers are an engineer's tool.
        self.details = ctk.CTkCheckBox(bar, text="Engineer details", width=130, command=self.on_details)
        self.details.pack(side="right", padx=(0, 10))
        self.hint = ctk.CTkLabel(bar, text="", text_color=DIM)
        self.hint.pack(side="right", padx=10)

        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.grid = ctk.CTkFrame(left, fg_color=VIDEO_BG, corner_radius=8)
        self.grid.pack(fill="both", expand=True)
        self.idle = ctk.CTkLabel(self.grid, text="Cameras stopped.\nTick a source and press Start.",
                                 text_color=OFF)
        self.idle.pack(expand=True)

        # Real-time performance, §2.6. One row per running camera plus the
        # machine's own numbers -- the answer to "is the pipeline keeping up".
        mon = ctk.CTkFrame(left, fg_color=PANEL)
        mon.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(mon, text="REAL-TIME PERFORMANCE", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.perf = ctk.CTkFrame(mon, fg_color="transparent")
        self.perf.pack(fill="x", padx=12, pady=(0, 10))
        self.sysline = ctk.CTkLabel(mon, text="", text_color=DIM, font=MONO, anchor="w")
        self.sysline.pack(anchor="w", padx=12, pady=(0, 8))

        right = ctk.CTkFrame(body, fg_color="transparent", width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        src = ctk.CTkFrame(right, fg_color=PANEL)
        src.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(src, text="CAMERAS", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(src, text="Tick every camera that watches this bottle.",
                     text_color=DIM, font=("Segoe UI", 13), wraplength=300,
                     justify="left").pack(anchor="w", padx=12)
        self.srcbox = ctk.CTkFrame(src, fg_color="transparent")
        self.srcbox.pack(fill="x", padx=12, pady=(6, 10))

        cap = ctk.CTkFrame(right, fg_color=PANEL)
        cap.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(cap, text="SNAPSHOT INTO A LABEL", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.snapfrom = ctk.CTkOptionMenu(cap, values=["-"], width=300)
        self.snapfrom.pack(padx=12, pady=(0, 6))
        self.capbox = ctk.CTkFrame(cap, fg_color="transparent")
        self.capbox.pack(fill="x", padx=12)
        ctk.CTkButton(cap, text="Capture frame", fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.capture).pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(cap, text="Tick nothing to capture a good bottle.", text_color=DIM,
                     font=("Segoe UI", 13)).pack(anchor="w", padx=12, pady=(0, 6))

        # A capture goes straight into training, so what was actually saved has
        # to be visible. A mis-aimed camera otherwise adds confident garbage to
        # the dataset and nothing on screen ever says so.
        self.last_capture: str | None = None
        self.shot = ctk.CTkFrame(cap, fg_color="transparent")
        self.shot.pack(fill="x", padx=12, pady=(0, 10))
        self.shot_img = ctk.CTkLabel(self.shot, text="", width=120, height=90)
        self.shot_img.pack(side="left")
        side = ctk.CTkFrame(self.shot, fg_color="transparent")
        side.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.shot_txt = ctk.CTkLabel(side, text="", text_color=DIM, justify="left",
                                     font=("Segoe UI", 13), wraplength=150, anchor="w")
        self.shot_txt.pack(anchor="w")
        self.undo_btn = ctk.CTkButton(side, text="Undo - delete it", height=26,
                                      fg_color="transparent", border_width=1, text_color=BAD,
                                      command=self.undo_capture)

        th = ctk.CTkFrame(right, fg_color=PANEL)
        th.pack(fill="both", expand=True)
        ctk.CTkLabel(th, text="THRESHOLDS", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.thbox = ctk.CTkScrollableFrame(th, fg_color="transparent")
        self.thbox.pack(fill="both", expand=True, padx=6)
        ctk.CTkLabel(th, text="Lower = catches more, fails more good bottles.",
                     text_color=DIM, font=("Segoe UI", 13), wraplength=300,
                     justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        self.scanned = False

    def refresh(self):
        for w in self.capbox.winfo_children():
            w.destroy()
        self.checks.clear()
        for d in self.app.defects:
            cb = ctk.CTkCheckBox(self.capbox, text=d, font=("Segoe UI", 14))
            cb.pack(anchor="w", pady=1)
            self.checks[d] = cb

        for w in self.thbox.winfo_children():
            w.destroy()
        self.sliders.clear()
        t = self.app.cfg.get("thresholds", {})
        for d in self.app.defects:
            box = ctk.CTkFrame(self.thbox, fg_color="transparent")
            box.pack(fill="x", pady=2)
            head = ctk.CTkFrame(box, fg_color="transparent")
            head.pack(fill="x")
            ctk.CTkLabel(head, text=d, font=("Segoe UI", 13)).pack(side="left")
            val = ctk.CTkLabel(head, text=f"{float(t.get(d, 0.5)):.2f}", font=("Segoe UI", 13, "bold"))
            val.pack(side="right")
            s = ctk.CTkSlider(box, from_=0.05, to=1.0, number_of_steps=19,
                              command=lambda v, d=d, lab=val: self.set_thr(d, v, lab))
            s.set(min(1.0, float(t.get(d, 0.5))))
            s.pack(fill="x")
            self.sliders[d] = s

        if not self.scanned:
            self.scanned = True
            self.scan()

    def set_thr(self, d, v, lab):
        lab.configure(text=f"{v:.2f}")
        cfg = D.load_config()
        cfg.setdefault("thresholds", {})[d] = round(float(v), 2)
        D.save_config(cfg)
        self.app.cfg = cfg

    # sources --------------------------------------------------------------
    def scan(self):
        if self.running:
            return messagebox.showinfo("Cameras running",
                                       "Stop the cameras before scanning for devices.")
        self.hint.configure(text="scanning…")
        n = int(self.app.settings.get("camera_probe", 5))
        self.app.run_bg(lambda: infer.list_cameras(n), self.set_sources)

    def set_sources(self, cams):
        keep = {k for k, cb in self.picked.items() if cb.get()}
        self.sources = [(f"Camera {c['index']} - {c['width']}x{c['height']}", c["index"])
                        for c in cams]
        up = D.DATA / "uploads"
        if up.exists():
            for p in sorted(up.iterdir()):
                if p.is_file():
                    self.sources.append((f"Video: {p.name}", str(p)))
        self.hint.configure(
            text="" if self.sources else "No camera found. Use “Open a video…” to test the model.")
        self.build_sources(keep or {infer.CameraSet.key(self.sources[0][1])} if self.sources else set())

    def build_sources(self, ticked: set):
        for w in self.srcbox.winfo_children():
            w.destroy()
        self.picked.clear()
        for name, val in self.sources:
            k = infer.CameraSet.key(val)
            cb = ctk.CTkCheckBox(self.srcbox, text=name, font=("Segoe UI", 14))
            cb.pack(anchor="w", pady=2)
            if k in ticked:
                cb.select()
            self.picked[k] = cb
        names = [n for n, _ in self.sources] or ["-"]
        self.snapfrom.configure(values=names)
        if self.snapfrom.get() not in names:
            self.snapfrom.set(names[0])

    def chosen(self) -> list:
        out = []
        for name, val in self.sources:
            cb = self.picked.get(infer.CameraSet.key(val))
            if cb is not None and cb.get():
                out.append((name, val))
        return out

    def pick_video(self):
        f = filedialog.askopenfilename(
            title="Choose a video to test the model on",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if not f:
            return
        cap = infer.open_capture(f)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            return messagebox.showerror(
                "Cannot read", f"OpenCV cannot decode:\n{Path(f).name}\n\nTry an MP4 (H.264).")
        # Play it where it sits; no copy into the project for a file already on
        # this machine. It only needs to be readable.
        self.sources.append((f"Video: {Path(f).name}", f))
        ticked = {k for k, cb in self.picked.items() if cb.get()}
        ticked.add(infer.CameraSet.key(f))
        self.build_sources(ticked)
        self.start()

    # run ------------------------------------------------------------------
    def start(self):
        chosen = self.chosen()
        if not chosen:
            return messagebox.showinfo("No source", "Tick at least one camera or open a video.")
        cfg = D.load_config()
        mode = self.infer_mode.get()
        stamp = cfg.get("active_model") if mode != "YOLO only" else None
        detector = None
        if mode != "Classifier":
            self.hint.configure(text="Loading the YOLO detector…")
            self.app.update_idletasks()
            try:
                detector = self.build_detector()
            except Exception as e:
                self.hint.configure(text="")
                return messagebox.showerror("YOLO detector", f"Not starting: {e}")
            self.hint.configure(text="")
        for name, val in chosen:
            self.app.cams.add(val, name)
        try:
            self.app.cams.start([v for _, v in chosen], stamp, detector)
        except Exception as e:
            messagebox.showwarning("Model", f"Running without a classifier: {e}")
            self.app.cams.start([v for _, v in chosen], None, detector)
        self.build_panes([n for n, _ in chosen])
        self.running = True
        self.app.cams.set_details(bool(self.details.get()))
        self._frames = {}
        self._render_gen = getattr(self, "_render_gen", 0) + 1
        threading.Thread(target=self._render_loop, args=(self._render_gen,), daemon=True).start()
        self.btn_start.configure(state="disabled")
        self.app.after(1200, self.check_started)
        self.tick()

    def _render_loop(self, gen):
        """Draw the annotated frames OFF the Tk thread (resize, overlay, colour convert), ~20 per second. The Tk
        thread only wraps the finished picture, so a slow overlay can no longer freeze the window or the buttons."""
        while self.running and gen == self._render_gen:
            t0 = time.monotonic()
            width = 880 if len(self.panes) <= 1 else 470
            for cam in list(self.app.cams.running()):
                try:
                    f = cam.overlay_frame(width=width)
                    if f is not None:
                        self._frames[cam.name] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                except Exception:                           # noqa: BLE001 - one bad frame must not stop the view
                    pass
            time.sleep(max(0.005, 0.05 - (time.monotonic() - t0)))

    def on_details(self):
        self.app.cams.set_details(bool(self.details.get()))

    def build_detector(self):
        """The shared YOLO detector, loaded on first use. Raises detect.DetectorError if the
        trained checkpoint is missing or is not the one recorded in MODEL_PROVENANCE.json.
        Confidence and weights path come from settings.json (keys detector_conf /
        detector_weights); the default confidence is a DEVELOPMENT value, not a validated one."""
        conf = float(self.app.settings.get("detector_conf", detect.DEV_CONF))
        weights = self.app.settings.get("detector_weights") or None
        recipe = D.load_config().get("inspection")
        bad = DEC.recipe_problems(recipe) if recipe else []
        if bad:
            raise detect.DetectorError("project inspection recipe (config.json 'inspection'): " + "; ".join(bad))
        need = DEC.recipe_classes(DEC.recipe_of({**DEC.RULES, "recipe": recipe}))
        d = self._detector
        if (d is None or d.conf != conf or str(d.weights) != str(weights or detect.DEFAULT_WEIGHTS)
                or getattr(d, "_require", None) != need):
            d = self._detector = detect.YoloDetector(weights=weights, conf=conf, require=need)
            d._require = need
        return d

    def build_panes(self, names):
        for w in self.grid.winfo_children():
            w.destroy()
        self.panes.clear()
        self._imgs.clear()
        cols = 1 if len(names) == 1 else 2
        for i, name in enumerate(names):
            cell = ctk.CTkFrame(self.grid, fg_color="transparent")
            cell.grid(row=i // cols, column=i % cols, sticky="nsew", padx=3, pady=3)
            # Plain tk.Label + PhotoImage: a CTkLabel re-wraps and re-scales its CTkImage on every configure(), which at
            # 20 frames/s per camera was most of the Live tab's main-thread time.
            lab = tk.Label(cell, text=name, fg=OFF, bg=VIDEO_BG, compound="center", bd=0, font=theme.BODY)
            lab.pack(fill="both", expand=True)
            self.panes[name] = lab
        for c in range(cols):
            self.grid.grid_columnconfigure(c, weight=1)
        for r in range((len(names) + cols - 1) // cols):
            self.grid.grid_rowconfigure(r, weight=1)

    def check_started(self):
        dead = [c for c in self.app.cams.cams.values() if c.error]
        if dead and not self.app.cams.running():
            self.stop()
            messagebox.showerror("Camera", dead[0].error)
        elif dead:
            self.hint.configure(text=f"{len(dead)} source failed: {dead[0].error}")

    def stop(self):
        self.running = False
        self.app.cams.stop()
        self.btn_start.configure(state="normal")
        self._imgs.clear()
        for w in self.grid.winfo_children():
            w.destroy()
        self.panes.clear()
        self.idle = ctk.CTkLabel(self.grid, text="Cameras stopped.\nTick a source and press Start.",
                                 text_color=OFF)
        self.idle.pack(expand=True)
        self.verdict.configure(text="")
        self.detlabel.configure(text="")
        for w in self.perf.winfo_children():
            w.destroy()
        self._perf_names = None

    def tick(self):
        if not self.running:
            return
        live = self.app.cams.running()
        for cam in live:
            lab = self.panes.get(cam.name)
            pil = self._frames.pop(cam.name, None)
            if lab is None or pil is None:
                continue
            img = ImageTk.PhotoImage(pil)
            self._imgs[cam.name] = img           # Tk drops unreferenced images
            lab.configure(image=img, text="")

        # Always refresh, even with no live camera: a dead camera must show FAULT,
        # not keep whatever verdict was on screen when it died.
        v = self.app.cams.display_verdict()                 # ONE stable verdict, in words (verdict.py)
        self.verdict.configure(text=v.text, text_color={VD.GOOD: GOOD, VD.DEFECT: BAD, VD.FAULT: WARN}.get(v.kind, DIM))
        parts = []
        for cam in (live if self.details.get() else ()):
            i = cam.inspection()
            if i.detector_state not in (None, infer.DET_OFF):
                n = (" " + " ".join(f"{k[0]}{v}" for k, v in i.detector.counts().items())
                     if i.detector else "")
                parts.append(f"{cam.name}: {i.detector_state}{n}"
                             + (f" ({i.detector_reason})" if i.detector_reason else ""))
        self.detlabel.configure(text=("Detector — " + " | ".join(parts)) if parts else "")
        self.app.after(50, self.tick)
        self.update_perf()

    def update_perf(self):
        hz = max(1, int(self.app.settings.get("monitor_hz", 2)))
        now = time.time()
        if now - getattr(self, "_perf_t", 0.0) < 1.0 / hz:
            return
        self._perf_t = now
        rows = self.app.cams.stats()
        # The table is built once per set of cameras and then only its text changes: destroying and recreating
        # ~15 CTk widgets every refresh was a visible stall on every update.
        names = tuple(s["name"] for s in rows)
        if names != getattr(self, "_perf_names", None):
            self._perf_names = names
            for w in self.perf.winfo_children():
                w.destroy()
            hdr = ("CAMERA", "FPS", "READ ms", "INFER ms", "LATENCY ms", "UNSCORED", "SCORED")
            head = ctk.CTkFrame(self.perf, fg_color="transparent")
            head.pack(fill="x")
            for t in hdr:
                ctk.CTkLabel(head, text=t, width=120, anchor="w", text_color=DIM,
                             font=("Segoe UI", 14, "bold")).pack(side="left")
            self._perf_cells = []
            for _ in rows:
                row = ctk.CTkFrame(self.perf, fg_color="transparent")
                row.pack(fill="x")
                self._perf_cells.append([ctk.CTkLabel(row, text="", width=120, anchor="w", font=MONO)
                                         for _ in hdr])
                for c in self._perf_cells[-1]:
                    c.pack(side="left")
        for s, cells in zip(rows, self._perf_cells):
            # A high unscored share means bottles can pass the camera without
            # the model ever looking at one of their frames.
            drop_col = BAD if s["drop_pct"] > 50 else WARN if s["drop_pct"] > 20 else DIM
            for c, (txt, col) in zip(cells, ((s["name"], INK), (s["fps"], INK), (s["read_ms"], DIM),
                                             (s["infer_ms"], INK), (s["latency_ms"], DIM),
                                             (f"{s['dropped']} ({s['drop_pct']}%)", drop_col),
                                             (s["scored"], DIM))):
                c.configure(text=str(txt), text_color=col)
        if now - getattr(self, "_sys_t", 0.0) < 1.0:       # CPU/RAM/GPU counters once a second is plenty
            return
        self._sys_t = now
        st = bench.system_stats()
        bits = []
        if st["cpu_sys"] is not None:
            bits.append(f"CPU {st['cpu_sys']:.0f}%")
        if st["cpu_proc"] is not None:
            bits.append(f"app {st['cpu_proc']:.0f}%")
        if st["ram_mb"]:
            bits.append(f"RAM {st['ram_mb']} MB")
        if st["gpu_name"]:
            g = f"GPU {st['gpu_name']}"
            if st["gpu_mem_mb"]:
                g += f"  {st['gpu_mem_mb']}/{st['gpu_mem_total_mb']} MB"
            if st["gpu_util"] is not None:
                g += f"  {st['gpu_util']}%"
            if st["gpu_temp"] is not None:
                g += f"  {st['gpu_temp']}°C"
            bits.append(g)
        self.sysline.configure(text="   ·   ".join(bits) or "no system counters available")

    def capture(self):
        name = self.snapfrom.get()
        cam = next((c for c in self.app.cams.running() if c.name == name), None)
        cam = cam or self.app.cams.primary()
        frame = cam.snapshot() if cam else None
        if frame is None:
            return messagebox.showinfo("No frame", "Start a camera first.")
        on = [d for d, cb in self.checks.items() if cb.get()]
        rel = D.save_capture(frame, on)
        for cb in self.checks.values():
            cb.deselect()

        self.last_capture = rel
        self._shot = bgr_to_ctk(frame, (120, 90))
        self.shot_img.configure(image=self._shot)
        self.shot_txt.configure(
            text=f"Saved from {cam.name} as\n{', '.join(on) if on else 'GOOD BOTTLE'}"
                 f"\n{Path(rel).name}",
            text_color=BAD if not on else WARN)
        self.undo_btn.pack(anchor="w", pady=(4, 0))
        self.app.reload()

    def undo_capture(self):
        rel = self.last_capture
        if not rel:
            return
        D.delete_images([rel])
        self.last_capture = None
        self.shot_img.configure(image=None)
        self.shot_txt.configure(text="Deleted.", text_color=DIM)
        self.undo_btn.pack_forget()
        self.app.reload()


# --------------------------------------------------------------- Data health
class DataTab:
    def __init__(self, app: App, parent):
        self.app = app
        left = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        roi = ctk.CTkFrame(left, fg_color=PANEL)
        roi.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(roi, text="CROP REGION (ROI)", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(roi, wraplength=700, justify="left", text_color=DIM,
                     text="Everything outside this box is thrown away before training: it removes "
                          "the empty background and normalises bottle scale. Leave generous margin "
                          "so a skewed bottle still fits. The box is stored with the frame size it "
                          "was measured on, so it rescales to a camera of another resolution."
                     ).pack(anchor="w", padx=14)
        f = ctk.CTkFrame(roi, fg_color="transparent")
        f.pack(anchor="w", padx=14, pady=8)
        self.roi_in = {}
        for k in ("x", "y", "w", "h"):
            ctk.CTkLabel(f, text=k).pack(side="left", padx=(8, 3))
            e = ctk.CTkEntry(f, width=80)
            e.pack(side="left")
            self.roi_in[k] = e
        self.roi_now = ctk.CTkLabel(roi, text="", text_color=DIM)
        self.roi_now.pack(anchor="w", padx=14)
        b = ctk.CTkFrame(roi, fg_color="transparent")
        b.pack(anchor="w", padx=14, pady=(6, 12))
        ctk.CTkButton(b, text="Save ROI", fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.save_roi).pack(side="left", padx=(0, 6))
        ctk.CTkButton(b, text="Re-measure", command=self.recalibrate).pack(side="left", padx=6)
        ctk.CTkButton(b, text="Preview crop", command=self.preview).pack(side="left", padx=6)
        self.preview_box = ctk.CTkLabel(roi, text="")
        self.preview_box.pack(anchor="w", padx=14, pady=(0, 12))

        dup = ctk.CTkFrame(left, fg_color=PANEL)
        dup.pack(fill="x")
        ctk.CTkLabel(dup, text="NEAR-DUPLICATE FRAMES", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(dup, wraplength=700, justify="left", text_color=DIM,
                     text="Consecutive captures of the same bottle. 300 images of 12 bottles is 12 "
                          "bottles' worth of information - usually the real ceiling on accuracy, "
                          "not the model or the epoch count."
                     ).pack(anchor="w", padx=14)
        ctk.CTkButton(dup, text="Scan", command=self.scan_dups).pack(anchor="w", padx=14, pady=8)
        self.dupout = ctk.CTkLabel(dup, text="", justify="left", anchor="w", font=MONO)
        self.dupout.pack(anchor="w", padx=14, pady=(0, 12))

        right = ctk.CTkFrame(parent, fg_color=PANEL, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="DATASET", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
        self.health = ctk.CTkFrame(right, fg_color="transparent")
        self.health.pack(fill="x", padx=14)

    def refresh(self):
        cfg = self.app.cfg
        r = cfg.get("roi") or [0, 0, 0, 0]
        for k, v in zip(("x", "y", "w", "h"), r):
            self.roi_in[k].delete(0, "end")
            self.roi_in[k].insert(0, str(int(v)))
        rf = cfg.get("roi_frame")
        self.roi_now.configure(
            text=f"Currently: {r if cfg.get('roi') else 'full frame'}"
                 + (f"   measured on {rf[0]}x{rf[1]}" if rf else "   (no frame size recorded)"))

        for w in self.health.winfo_children():
            w.destroy()
        c = self.app.counts
        rows = [("Total images", c["_total"], None),
                ("GOOD (passes)", c.get("_good", 0), WARN if c.get("_good", 0) < 300 else GOOD),
                ("DEFECTIVE (fails)", c.get("_defective", 0), None),
                ("Unreviewed", c.get("_unreviewed", 0), WARN if c.get("_unreviewed") else None),
                ("", "", None)]
        rows += [(d, c.get(d, 0), BAD if not c.get(d) else WARN if c.get(d, 0) < 40 else None)
                 for d in self.app.defects]
        for name, val, col in rows:
            row = ctk.CTkFrame(self.health, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=name, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=str(val), text_color=col,
                         font=("Segoe UI", 14, "bold")).pack(side="right")

    def save_roi(self):
        try:
            v = [int(float(self.roi_in[k].get())) for k in ("x", "y", "w", "h")]
        except ValueError:
            return messagebox.showerror("ROI", "All four values must be numbers.")
        cfg = D.load_config()
        cfg["roi"] = v if v[2] > 0 and v[3] > 0 else None
        D.save_config(cfg)
        self.app.reload()
        messagebox.showinfo("ROI saved",
                            "The crop cache is keyed on the ROI, so the next training run re-caches.")

    def recalibrate(self):
        def work():
            import calibrate
            return calibrate.run()
        self.app.run_bg(work, lambda roi: (self.app.reload(),
                                           messagebox.showinfo("Measured", f"ROI = {roi}")))

    def preview(self):
        paths = sorted(self.app.labels, key=D._natkey)
        if not paths:
            return
        rel = paths[0]
        img = D.imread(D.IMAGE_ROOT / rel)
        if img is None:
            return
        cfg = D.load_config()
        crop = D.model_input(img, cfg)
        h, w = img.shape[:2]
        s = 380 / max(h, w)
        shown = cv2.resize(img, (round(w * s), round(h * s)))
        if cfg.get("roi"):
            rx, ry, rw, rh = cfg["roi"]
            rf = cfg.get("roi_frame")
            if rf and (rf[0], rf[1]) != (w, h):
                fx, fy = w / rf[0], h / rf[1]
                rx, ry, rw, rh = rx * fx, ry * fy, rw * fx, rh * fy
            cv2.rectangle(shown, (round(rx * s), round(ry * s)),
                          (round((rx + rw) * s), round((ry + rh) * s)), (90, 220, 90), 2)
        pad = np.zeros((max(shown.shape[0], crop.shape[0]),
                        shown.shape[1] + crop.shape[1] + 12, 3), np.uint8)
        pad[:shown.shape[0], :shown.shape[1]] = shown
        pad[:crop.shape[0], shown.shape[1] + 12:] = crop
        self._p = bgr_to_ctk(pad)
        self.preview_box.configure(image=self._p, text="")

    def scan_dups(self):
        self.dupout.configure(text="Scanning…")

        def work():
            paths = sorted(self.app.labels, key=D._natkey)
            sm = D.scene_map(paths)
            runs: dict[str, list[str]] = {}
            for p in paths:
                runs.setdefault(sm.get(p, p), []).append(p)
            return len(paths), runs

        def done(r):
            n, runs = r
            big = sorted((v for v in runs.values() if len(v) > 1), key=len, reverse=True)[:14]
            txt = f"{n} images  ->  ~{len(runs)} distinct scenes\n\n"
            txt += "\n".join(f"  {v[0].rsplit('/', 1)[0]:<20} run of {len(v):<4} "
                             f"from {v[0].rsplit('/', 1)[-1]}" for v in big)
            self.dupout.configure(text=txt)
        self.app.run_bg(work, done)


# ------------------------------------------------------------------ Analysis
class AnalysisTab:
    """Everything about a trained model after the run: curves, per-defect
    scores, the confusion matrix, and a like-for-like comparison of every
    checkpoint the project holds."""

    def __init__(self, app: App, parent):
        self.app = app
        self.metrics: dict | None = None
        self.compare_rows: list[dict] = []

        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(bar, text="Model").pack(side="left", padx=(12, 6), pady=8)
        self.pick = ctk.CTkOptionMenu(bar, values=["-"], width=210, command=lambda _: self.load())
        self.pick.pack(side="left")
        ctk.CTkButton(bar, text="Make active", width=110, fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.activate).pack(side="left", padx=8)
        ctk.CTkLabel(bar, text="Defect").pack(side="left", padx=(18, 6))
        self.defect = ctk.CTkOptionMenu(bar, values=["-"], width=170,
                                        command=lambda _: self.draw_defect())
        self.defect.pack(side="left")
        self.btn_cmp = ctk.CTkButton(bar, text="Re-test every model on today's labels",
                                     command=self.compare)
        self.btn_cmp.pack(side="right", padx=12)

        body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self.body = body

        self.summary = ctk.CTkLabel(body, text="", text_color=DIM, justify="left", anchor="w")
        self.summary.pack(fill="x", pady=(0, 6))

        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 8))
        self.c_loss = chart_panel(row1, "TRAINING LOSS PER EPOCH", 470, 230,
                                  "Falling and flattening is what you want. Still falling at the "
                                  "last epoch means it was stopped early.")
        self.c_f1 = chart_panel(row1, "VALIDATION MACRO-F1 PER EPOCH", 470, 230,
                                "The checkpoint kept is the best epoch, not the last one.")

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 8))
        self.c_bars = chart_panel(row2, "PER-DEFECT SCORES (best epoch)", 470, 230,
                                  "Recall is the number that matters - a missed defect ships.")
        self.c_conf = chart_panel(row2, "CONFUSION MATRIX (selected defect)", 470, 230,
                                  "Multi-label means one 2x2 per defect: every defect is its own "
                                  "yes/no question about the same bottle.")

        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.pack(fill="x", pady=(0, 8))
        self.c_recall = chart_panel(row3, "RECALL PER DEFECT, PER EPOCH", 950, 240,
                                    "A macro average hides one class collapsing while the rest "
                                    "improve. This is where that shows up.")

        ctk.CTkLabel(body, text="MODEL COMPARISON", text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8, 2))
        self.cmp_note = ctk.CTkLabel(
            body, text_color=DIM, justify="left", wraplength=700,
            text="STORED F1 is each model's own validation run, measured against the labels as "
                 "they stood that day - after any relabelling those numbers are no longer "
                 "comparable. TODAY F1 re-scores each model against current labels, on the exact "
                 "validation images that model was held out from. A checkpoint trained before "
                 "that set was recorded shows “needs retrain”: scoring it would mean feeding it "
                 "images it memorised, which returns a perfect 1.000 for every defect and means "
                 "nothing.")
        self.cmp_note.pack(anchor="w", pady=(0, 6))
        self.cmp = ctk.CTkFrame(body, fg_color="transparent")
        self.cmp.pack(fill="x")

    # ------------------------------------------------------------------ data
    def refresh(self):
        stamps = infer.list_models()
        self.pick.configure(values=stamps or ["-"])
        active = self.app.cfg.get("active_model")
        want = active if active in stamps else (stamps[0] if stamps else "-")
        self.pick.set(want)
        self.load()
        self.show_compare()

    def load(self):
        stamp = self.pick.get()
        self.metrics = None
        p = D.MODELS / stamp / "metrics.json" if stamp and stamp != "-" else None
        if p and p.exists():
            try:
                self.metrics = json.loads(p.read_text())
            except Exception:
                self.metrics = None
        m = self.metrics
        if not m:
            self.summary.configure(text="No trained model yet. Train one from the Train tab.")
            for cv in (self.c_loss, self.c_f1, self.c_bars, self.c_conf, self.c_recall):
                cv.delete("all")
            self.defect.configure(values=["-"])
            self.defect.set("-")
            return

        scored = [d for d in m["defects"] if m["per_defect"][d]["n_pos"]]
        self.defect.configure(values=scored or ["-"])
        if self.defect.get() not in scored:
            self.defect.set(scored[0] if scored else "-")
        self.summary.configure(
            text=f"{m['stamp']}   ·   macro-F1 {m['macro_f1']}   ·   {m.get('arch', 'efficientnet_b0')}   ·   "
                 f"{m['n_train']} train / {m['n_val']} val images"
                 + (f" (~{m['n_scenes']} scenes)" if m.get("n_scenes") else "")
                 + f"   ·   {m['epochs']} epochs"
                 + (f"   ·   {m['device']}" if m.get("device") else "")
                 + ("" if m.get("val_paths") else
                    "   ·   predates validation-set recording, so it cannot be re-tested"))
        self.draw()

    def draw(self):
        m = self.metrics
        if not m:
            return
        hist = m.get("history") or []
        if hist:
            xs = [h["epoch"] for h in hist]
            charts.line_chart(self.c_loss, {"loss": [h["loss"] for h in hist]},
                              x_values=xs, legend=False)
            charts.line_chart(self.c_f1, {"macro-F1": [h["macro_f1"] for h in hist]},
                              x_values=xs, y_lo=0, y_hi=1, legend=False)
            per = {d: [h["recall"].get(d, 0.0) for h in hist]
                   for d in m["defects"] if any(h["recall"].get(d) for h in hist)}
            charts.line_chart(self.c_recall, per, x_values=xs, y_lo=0, y_hi=1)
        else:
            for cv in (self.c_loss, self.c_f1, self.c_recall):
                cv.delete("all")
                cv.create_text(140, 40, text="trained before curves were recorded",
                               fill=DIM, anchor="w")

        scored = [d for d in m["defects"] if m["per_defect"][d]["n_pos"]]
        if scored:
            charts.bar_chart(self.c_bars, [d[:12] for d in scored],
                             [m["per_defect"][d]["recall"] for d in scored],
                             colours=[charts.GREEN if m["per_defect"][d]["recall"] >= 0.9
                                      else charts.AMBER if m["per_defect"][d]["recall"] >= 0.7
                                      else charts.RED for d in scored],
                             y_hi=1.0)
        self.draw_defect()

    def draw_defect(self):
        m, d = self.metrics, self.defect.get()
        if not m or d not in m.get("per_defect", {}):
            return
        x = m["per_defect"][d]
        tp, fp, fn = x.get("tp", 0), x.get("fp", 0), x.get("fn", 0)
        tn = max(0, m.get("n_val", tp + fp + fn) - tp - fp - fn)
        charts.confusion(self.c_conf, tp, fp, fn, tn)

    def activate(self):
        stamp = self.pick.get()
        if stamp and stamp != "-":
            self.app.tab_train.activate(stamp)

    # ------------------------------------------------------------- comparison
    def compare(self):
        stamps = infer.list_models()
        if not stamps:
            return messagebox.showinfo("No models", "Train a model first.")
        if not messagebox.askyesno(
                "Re-test every model",
                f"Score all {len(stamps)} checkpoint(s) against today's labels?\n\n"
                f"This runs each model over the validation split - about "
                f"{len(stamps) * 3}-{len(stamps) * 15} seconds."):
            return
        self.btn_cmp.configure(state="disabled", text="Re-testing…")

        def work():
            import train
            out = []
            for s in stamps:
                try:
                    out.append(train.score_checkpoint(s, log=lambda _m: None))
                except Exception as e:
                    out.append({"stamp": s, "error": f"{type(e).__name__}: {e}"})
            return out

        def done(rows):
            self.compare_rows = rows
            self.btn_cmp.configure(state="normal", text="Re-test every model on today's labels")
            self.show_compare()
        self.app.run_bg(work, done)

    def show_compare(self):
        for w in self.cmp.winfo_children():
            w.destroy()
        stamps = infer.list_models()
        if not stamps:
            return
        retested = {r["stamp"]: r for r in self.compare_rows}
        cols = (("MODEL", 165), ("WHEN", 60), ("EPOCHS", 70), ("VAL", 60),
                ("STORED F1", 90), ("TODAY F1", 90), ("WORST DEFECT", 210), ("", 90))
        hdr = ctk.CTkFrame(self.cmp, fg_color="transparent")
        hdr.pack(fill="x")
        for t, w in cols:
            ctk.CTkLabel(hdr, text=t, width=w, anchor="w", text_color=DIM,
                         font=("Segoe UI", 12, "bold")).pack(side="left")
        active = self.app.cfg.get("active_model")
        best = max((r.get("macro_f1", 0) for r in retested.values()), default=None)
        for s in stamps:
            p = D.MODELS / s / "metrics.json"
            m = {}
            if p.exists():
                try:
                    m = json.loads(p.read_text())
                except Exception:
                    m = {}
            r = retested.get(s, {})
            row = ctk.CTkFrame(self.cmp, fg_color=BG if s != active else ACC_SOFT)
            row.pack(fill="x", pady=1)
            worst, wr = "-", None
            src = r.get("per_defect") or m.get("per_defect") or {}
            scored = {d: v for d, v in src.items() if v.get("n_pos")}
            if scored:
                worst = min(scored, key=lambda d: scored[d]["recall"])
                wr = scored[worst]["recall"]
            today = r.get("macro_f1")
            vals = ((s + ("  (active)" if s == active else ""), 165, INK),
                    (s[4:8] if len(s) > 8 else "", 60, DIM),
                    (m.get("epochs", "-"), 70, DIM),
                    (r.get("n_val") or m.get("n_val", "-"), 60, DIM),
                    (m.get("macro_f1", "-"), 90, DIM),
                    ("needs retrain" if r.get("error") else
                     (today if today is not None else "-"), 90,
                     WARN if r.get("error") else
                     (GOOD if today is not None and best and today >= best else INK)),
                    (f"{worst} {wr:.2f}" if wr is not None else "-", 210,
                     BAD if wr is not None and wr < 0.7 else DIM))
            for txt, w, col in vals:
                ctk.CTkLabel(row, text=str(txt), width=w, anchor="w", text_color=col,
                             font=MONO).pack(side="left", pady=3)
            ctk.CTkButton(row, text="Activate", width=84, height=24, fg_color="transparent",
                          border_width=1, text_color=ACC, command=lambda s=s: self.app.tab_train.activate(s)
                          ).pack(side="left", padx=4)


# ------------------------------------------------------------------- Camera
class BenchTab:
    """Camera benchmarking, §2.7: what the camera actually delivers at each
    setting, rather than what it was asked for."""

    def __init__(self, app: App, parent):
        self.app = app
        self.rows: list[dict] = []
        self._stop = False

        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(bar, text="Camera").pack(side="left", padx=(12, 6), pady=8)
        self.cam = ctk.CTkOptionMenu(bar, values=["-"], width=220)
        self.cam.pack(side="left")
        ctk.CTkButton(bar, text="⟳ Scan", width=80, command=self.scan).pack(side="left", padx=6)
        ctk.CTkLabel(bar, text="Seconds each").pack(side="left", padx=(16, 6))
        self.secs = ctk.CTkEntry(bar, width=60)
        self.secs.insert(0, str(app.settings.get("bench_seconds", 3.0)))
        self.secs.pack(side="left")
        self.use_model = ctk.CTkCheckBox(bar, text="also time the active model")
        self.use_model.select()
        self.use_model.pack(side="left", padx=14)
        self.btn = ctk.CTkButton(bar, text="Run benchmark", fg_color=ACC, text_color=ACC_T,
                                 hover_color=ACC_H, command=self.run)
        self.btn.pack(side="left", padx=8)
        ctk.CTkButton(bar, text="Export CSV", command=self.export).pack(side="right", padx=12)

        # Lock the camera for the line: settings.json "camera_controls" (infer.open_capture applies it
        # every time the camera opens). Blank = leave that one on auto. Values are the driver's own scale.
        lock = ctk.CTkFrame(parent, fg_color=PANEL)
        lock.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(lock, text="Lock camera").pack(side="left", padx=(12, 10), pady=8)
        self.ctl = {}
        for key, label in (("focus", "Focus"), ("exposure", "Exposure"), ("wb_temperature", "WB K")):
            ctk.CTkLabel(lock, text=label).pack(side="left", padx=(6, 4))
            e = ctk.CTkEntry(lock, width=64)
            e.pack(side="left")
            self.ctl[key] = e
        ctk.CTkLabel(lock, text="Rotate").pack(side="left", padx=(12, 4))
        self.rot = ctk.CTkOptionMenu(lock, values=["0", "90", "180", "270"], width=70)
        self.rot.pack(side="left")
        ctk.CTkButton(lock, text="Save + read back", command=self.save_controls).pack(side="left", padx=10)
        self.ctl_msg = ctk.CTkLabel(lock, text="", font=MONO, text_color=DIM, justify="left")
        self.ctl_msg.pack(side="left", padx=6)

        ctk.CTkLabel(parent, text_color=DIM, justify="left", wraplength=700,
                     text="A webcam asked for 1920x1080 at 30 fps will quietly deliver 7 in poor "
                          "light, and OpenCV reports the number it was asked for. Every column "
                          "here is measured from real frames. Sharpness is the variance of the "
                          "Laplacian: it is what decides whether a meniscus line or label print "
                          "is resolvable at all, which no frame rate can compensate for."
                     ).pack(anchor="w", pady=(0, 8))

        self.out = ctk.CTkFrame(parent, fg_color="transparent")
        self.out.pack(fill="x")
        self.chart = ctk.CTkCanvas(parent, height=210, bg=PANEL, highlightthickness=0)
        self.chart.pack(fill="x", pady=8)
        self.log = ctk.CTkTextbox(parent, height=120, font=MONO, fg_color=PANEL,
                                  text_color=INK)
        self.log.pack(fill="both", expand=True)
        self.scanned = False

    def refresh(self):
        if not self.scanned:
            self.scanned = True
            self.scan()

    def scan(self):
        n = int(self.app.settings.get("camera_probe", 5))
        self.app.run_bg(lambda: infer.list_cameras(n), self.set_cams)

    def set_cams(self, cams):
        names = [f"Camera {c['index']} - {c['width']}x{c['height']}" for c in cams] or ["-"]
        self._idx = {n: c["index"] for n, c in zip(names, cams)}
        self.cam.configure(values=names, command=lambda _: self.load_controls())
        self.cam.set(names[0])
        self.load_controls()

    def load_controls(self):
        idx = getattr(self, "_idx", {}).get(self.cam.get())
        c = (self.app.settings.get("camera_controls") or {}).get(str(idx), {}) if idx is not None else {}
        for k, e in self.ctl.items():
            e.delete(0, "end")
            if k in c:
                e.insert(0, str(c[k]))
        self.rot.set(str(c.get("rotate", 0)))

    def _controls_from_ui(self) -> dict:
        """The camera_controls entry the fields describe. A manual value switches its auto mode off
        (auto_exposure 0.25 is OpenCV's 'manual' value); a blank field leaves that control on auto."""
        out = {}
        for k, e in self.ctl.items():
            t = e.get().strip()
            if not t:
                continue
            try:
                out[k] = float(t)
            except ValueError:
                raise ValueError(f"{k}: {t!r} is not a number")
        if "focus" in out:
            out["autofocus"] = 0
        if "exposure" in out:
            out["auto_exposure"] = 0.25
        if "wb_temperature" in out:
            out["auto_wb"] = 0
        if int(self.rot.get()):
            out["rotate"] = int(self.rot.get())
        return out

    def save_controls(self):
        idx = getattr(self, "_idx", {}).get(self.cam.get())
        if idx is None:
            return messagebox.showinfo("No camera", "Scan and pick a camera first.")
        if self.app.cams.running():
            return messagebox.showinfo("Cameras running", "Stop the Live tab first - the camera has to be "
                                                          "reopened to apply the settings.")
        try:
            c = self._controls_from_ui()
        except ValueError as e:
            return self.ctl_msg.configure(text=str(e), text_color=BAD)
        allc = dict(self.app.settings.get("camera_controls") or {})
        if c:
            allc[str(idx)] = c
        else:
            allc.pop(str(idx), None)
        self.app.settings["camera_controls"] = allc
        D.save_settings(self.app.settings)
        self.ctl_msg.configure(text="saved, reading back…", text_color=DIM)

        def work():
            cap = infer.open_capture(idx)
            try:
                for _ in range(5):
                    cap.read()
            finally:
                cap.release()
            return infer.applied_controls.get(str(idx)) or {}

        def done(got):
            ignored = [k for k, (want, have) in got.items() if have is None or abs(float(want) - have) > 0.01]
            text = "  ".join(f"{k} {w}->{h}" for k, (w, h) in got.items()) or "all on auto"
            self.ctl_msg.configure(text=text + (f"\nIGNORED by the driver: {', '.join(ignored)}" if ignored else ""),
                                   text_color=WARN if ignored else GOOD)
        self.app.run_bg(work, done)

    def run(self):
        idx = getattr(self, "_idx", {}).get(self.cam.get())
        if idx is None:
            return messagebox.showinfo("No camera", "Scan and pick a camera first.")
        if self.app.cams.running():
            return messagebox.showinfo(
                "Cameras running",
                "Stop the Live tab first - a camera can only be opened by one thing at a time.")
        try:
            secs = float(self.secs.get())
        except ValueError:
            return messagebox.showerror("Seconds", "Seconds must be a number.")

        stamp = self.app.cfg.get("active_model") if self.use_model.get() else None
        self.btn.configure(state="disabled", text="Running…")
        self.log.delete("1.0", "end")
        self._stop = False

        def emit(line):
            self.app.post(lambda: (self.log.insert("end", str(line) + "\n"), self.log.see("end")))

        def work():
            model = None
            if stamp:
                try:
                    model = infer.Model(stamp)
                except Exception as e:
                    emit(f"(no model timing: {e})")
            return bench.benchmark_camera(idx, seconds=secs, model=model, progress=emit,
                                          should_stop=lambda: self._stop)

        def done(rows):
            self.rows = rows
            self.btn.configure(state="normal", text="Run benchmark")
            self.show()
            b = bench.best_combo(rows)
            emit("")
            emit(f"recommended: {b['requested']} - sharpest configuration that held its "
                 f"frame rate and resolution" if b else
                 "no configuration delivered what it was asked for - check lighting and the cable")
        self.app.run_bg(work, done)

    def show(self):
        for w in self.out.winfo_children():
            w.destroy()
        if not self.rows:
            ctk.CTkLabel(self.out, text_color=DIM, justify="left",
                         text="No benchmark run yet. Pick a camera and press Run benchmark — "
                              "it sweeps five configurations, a few seconds each.").pack(anchor="w")
            return
        cols = (("REQUESTED", 130), ("ACTUAL", 110), ("FPS", 70), ("LATENCY ms", 100),
                ("JITTER ms", 90), ("SHARPNESS", 100), ("BRIGHT", 80), ("CLIPPED %", 90),
                ("INFER ms", 90))
        hdr = ctk.CTkFrame(self.out, fg_color="transparent")
        hdr.pack(fill="x")
        for t, w in cols:
            ctk.CTkLabel(hdr, text=t, width=w, anchor="w", text_color=DIM,
                         font=("Segoe UI", 12, "bold")).pack(side="left")
        best = bench.best_combo(self.rows)
        for r in self.rows:
            row = ctk.CTkFrame(self.out,
                               fg_color=ACC_SOFT if best and r is best else BG)
            row.pack(fill="x", pady=1)
            if r.get("error"):
                ctk.CTkLabel(row, text=f"{r['requested']}   {r['error']}", anchor="w",
                             text_color=BAD, font=MONO).pack(side="left", pady=3)
                continue
            want_fps = float(r["requested"].split("@")[1])
            vals = ((r["requested"], 130, INK),
                    (r["actual_size"], 110, INK if r["size_ok"] else BAD),
                    (r["fps"], 70, GOOD if r["fps"] >= 0.8 * want_fps else BAD),
                    (r["latency_ms"], 100, DIM), (r["jitter_ms"], 90, DIM),
                    (r["sharpness"], 100, INK),
                    (r["brightness"], 80, WARN if (r["brightness"] or 0) < 40 else DIM),
                    (r["clipped"], 90, WARN if (r["clipped"] or 0) > 2 else DIM),
                    (r.get("infer_ms") or "-", 90, DIM))
            for txt, w, col in vals:
                ctk.CTkLabel(row, text=str(txt), width=w, anchor="w", text_color=col,
                             font=MONO).pack(side="left", pady=3)
        ok = [r for r in self.rows if not r.get("error") and r.get("sharpness")]
        if ok:
            charts.bar_chart(self.chart, [r["requested"].replace("1920x1080", "1080p") for r in ok],
                             [r["sharpness"] for r in ok],
                             colours=[charts.BLUE if r is not best else charts.GREEN for r in ok])

    def export(self):
        if not self.rows:
            return messagebox.showinfo("Nothing to export", "Run a benchmark first.")
        f = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="camera_benchmark.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not f:
            return
        keys = sorted({k for r in self.rows for k in r})
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        messagebox.showinfo("Exported", f"{len(self.rows)} rows written to\n{f}")


# ------------------------------------------------------------------ Settings
class SettingsTab:
    def __init__(self, app: App, parent):
        self.app = app
        wrap = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        wrap.pack(fill="both", expand=True)

        look = self._box(wrap, "APPEARANCE")
        ctk.CTkLabel(look, text="Text and control size", anchor="w").pack(anchor="w", padx=14)
        rowf = ctk.CTkFrame(look, fg_color="transparent")
        rowf.pack(fill="x", padx=14, pady=(2, 2))
        self.font_val = ctk.CTkLabel(rowf, text="", width=60, font=("Segoe UI", 14, "bold"))
        self.font_val.pack(side="right")
        self.font = ctk.CTkSlider(rowf, from_=0.8, to=1.6, number_of_steps=16,
                                  command=self.preview_font)
        self.font.set(float(app.settings.get("font_scale", 1.0)))
        self.font.pack(side="left", fill="x", expand=True)
        # Rescaling every widget is slow, so it happens once, when the slider is let go -- not on every step of a drag
        self.font.bind("<ButtonRelease-1>", lambda e: self.app.after(30, lambda: self.set_font(self.font.get())))
        ctk.CTkLabel(look, text_color=DIM, justify="left", wraplength=700,
                     text="Scales every control with the text, so buttons grow with their labels "
                          "instead of clipping them. Saved when you let go of the slider; applied the next time the app starts."
                     ).pack(anchor="w", padx=14, pady=(0, 4))
        quick = ctk.CTkFrame(look, fg_color="transparent")
        quick.pack(anchor="w", padx=14, pady=(0, 12))
        self.restart_btn = ctk.CTkButton(quick, text="Restart now", width=110, fg_color=ACC, text_color=ACC_T,
                                         hover_color=ACC_H, state="disabled", command=self.app.restart,
                                         text_color_disabled=DIM)
        self.restart_btn.configure(fg_color=PANEL_2)
        self.restart_note = ctk.CTkLabel(quick, text="In use now.", text_color=DIM)
        for lab, v in (("Small", 0.9), ("Normal", 1.0), ("Large", 1.2), ("Very large", 1.45)):
            ctk.CTkButton(quick, text=lab, width=90, fg_color="transparent", border_width=1, text_color=ACC,
                          command=lambda v=v: self.set_font(v)
                          ).pack(side="left", padx=(0, 6))
        self.restart_btn.pack(side="left", padx=(14, 8))
        self.restart_note.pack(side="left")
        trow = ctk.CTkFrame(look, fg_color="transparent")
        trow.pack(anchor="w", padx=14, pady=(0, 4))
        ctk.CTkLabel(trow, text="Theme", width=120, anchor="w").pack(side="left")
        self.theme = ctk.CTkSegmentedButton(trow, values=["light", "dark"], command=self.set_theme)
        self.theme.set("dark" if app.settings.get("ui_theme") == "dark" else "light")
        self.theme.pack(side="left")
        ctk.CTkLabel(look, text_color=DIM, justify="left", wraplength=700,
                     text="Light industrial HMI by default (applied at the next start). Red, amber and green are kept "
                          "only where they carry meaning - a reject an operator reads across a room, a warning, a "
                          "passing recall - so the theme cannot make a reject look like a pass. Camera images "
                          "always sit on a dark background."
                     ).pack(anchor="w", padx=14, pady=(0, 12))

        prod = self._box(wrap, "PRODUCTION")
        prow = ctk.CTkFrame(prod, fg_color="transparent")
        prow.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkLabel(prow, text="Evidence images kept", width=330, anchor="w").pack(side="left")
        self.evidence = ctk.CTkOptionMenu(prow, values=list(production_store.EVIDENCE_POLICIES), width=180)
        self.evidence.set(str(app.settings.get("evidence_policy", "REJECT_AND_FAULT")))
        self.evidence.pack(side="left")
        ctk.CTkLabel(prod, text="Every bottle is always recorded (production.db); this decides which ones also keep "
                                "an image. SAMPLE:10 = every 10th bottle plus every REJECT / FAULT.", text_color=DIM,
                     font=("Segoe UI", 12), wraplength=700, justify="left").pack(anchor="w", padx=14, pady=(0, 8))
        self.sim_chk = ctk.CTkCheckBox(prod, text="Show the engineer 'Simulation check' (ladder + code checks; applies at next start)")
        if app.settings.get("show_simulation_check", True):
            self.sim_chk.select()
        self.sim_chk.pack(anchor="w", padx=14, pady=(0, 4))
        self.auto_act = ctk.CTkCheckBox(prod, text="Activate a newly trained classifier automatically (NOT recommended)")
        if app.settings.get("auto_activate_trained_model"):
            self.auto_act.select()
        self.auto_act.pack(anchor="w", padx=14, pady=(0, 4))
        ctk.CTkLabel(prod, text="Off: training registers a CANDIDATE; it reaches production only through Models "
                                "(validate on the real camera, approve, activate) with a rollback record.",
                     text_color=DIM, font=("Segoe UI", 12), wraplength=700, justify="left").pack(anchor="w", padx=14,
                                                                                               pady=(0, 8))
        self.pin = self._number(prod, "Engineer PIN (blank = none)", app.settings.get("engineer_pin", ""),
                                "Asked when switching from OPERATOR to ENGINEER mode. A convenience lock, not "
                                "security: settings.json is readable on this PC.")

        cam = self._box(wrap, "CAMERAS AND MONITORING")
        self.probe = self._number(cam, "Camera indices to probe when scanning",
                                  app.settings.get("camera_probe", 5),
                                  "Higher finds cameras on high indices; each extra index costs "
                                  "about a second per scan.")
        self.bsecs = self._number(cam, "Benchmark seconds per configuration",
                                  app.settings.get("bench_seconds", 3.0),
                                  "Under about 2 s the frame-rate estimate is mostly noise.")
        self.hz = self._number(cam, "Performance panel refreshes per second",
                               app.settings.get("monitor_hz", 2),
                               "The panel rebuilds its rows each refresh; above about 4 Hz it "
                               "competes with the video for the main thread.")

        lab = self._box(wrap, "LABELLING")
        self.per_page = self._number(lab, "Thumbnails per page",
                                     app.settings.get("per_page", 60),
                                     "Bigger pages mean fewer clicks and slower page loads.")

        btns = ctk.CTkFrame(wrap, fg_color="transparent")
        btns.pack(fill="x", pady=10)
        ctk.CTkButton(btns, text="Save settings", fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.save).pack(side="left")
        ctk.CTkButton(btns, text="Reset to defaults", fg_color="transparent", border_width=1, text_color=ACC,
                      command=self.reset).pack(side="left", padx=8)
        self.saved = ctk.CTkLabel(btns, text="", text_color=GOOD)
        self.saved.pack(side="left", padx=12)

        where = self._box(wrap, "WHERE THINGS ARE")
        self.paths = ctk.CTkLabel(where, text="", justify="left", anchor="w",
                                  font=MONO, text_color=DIM)
        self.paths.pack(anchor="w", padx=14, pady=(0, 12))

    def _box(self, parent, title):
        f = ctk.CTkFrame(parent, fg_color=PANEL)
        f.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f, text=title, text_color=DIM,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        return f

    def _number(self, parent, label, value, note):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkLabel(row, text=label, anchor="w", width=330).pack(side="left")
        e = ctk.CTkEntry(row, width=90)
        e.insert(0, str(value))
        e.pack(side="left")
        ctk.CTkLabel(parent, text=note, text_color=DIM, font=("Segoe UI", 12),
                     wraplength=700, justify="left").pack(anchor="w", padx=14, pady=(0, 10))
        return e

    def refresh(self):
        self.font_val.configure(text=f"{float(self.font.get()):.2f}x")
        self.paths.configure(
            text=f"project    {D.PROJECT_DIR}\n"
                 f"images     {D.IMAGE_ROOT}\n"
                 f"labels     {D.LABELS_CSV}\n"
                 f"models     {D.MODELS}\n"
                 f"settings   {D.SETTINGS_JSON}")

    def preview_font(self, v):
        self.font_val.configure(text=f"{float(v):.2f}x")

    def set_theme(self, v):
        if self.app.settings.get("ui_theme", "light") != v:
            self.app.settings["ui_theme"] = v
            D.save_settings(self.app.settings)
        pending = v != theme.MODE
        self.restart_note.configure(text="Saved. Press Restart now to apply." if pending else "In use now.",
                                    text_color=WARN if pending else DIM)
        self.restart_btn.configure(state="normal" if pending else "disabled", fg_color=ACC if pending else PANEL_2,
                                   text_color=ACC_T if pending else DIM)

    def set_font(self, v):
        """Save the text size; it takes effect at the next start. Re-scaling every widget in a running window means
        redrawing ~3,000 of them (20+ s on this PC, during which the app looked frozen), so it is done once, at
        start-up, when none exist yet."""
        v = round(max(0.7, min(1.8, float(v))), 2)
        self.font.set(v)
        pending = abs(v - self.app.applied_scale) > 1e-6
        self.font_val.configure(text=f"{v:.2f}x")
        self.restart_note.configure(text="Saved. Press Restart now to apply." if pending else "In use now.",
                                    text_color=WARN if pending else DIM)
        self.restart_btn.configure(state="normal" if pending else "disabled", fg_color=ACC if pending else PANEL_2,
                                   text_color=ACC_T if pending else DIM)
        if abs(v - float(self.app.settings.get("font_scale", 1.0))) > 1e-6:
            self.app.settings["font_scale"] = v
            D.save_settings(self.app.settings)

    def save(self):
        s = self.app.settings
        for key, widget, cast in (("camera_probe", self.probe, int),
                                  ("bench_seconds", self.bsecs, float),
                                  ("monitor_hz", self.hz, int),
                                  ("per_page", self.per_page, int)):
            try:
                s[key] = cast(widget.get())
            except ValueError:
                return messagebox.showerror("Settings", f"{key} must be a number.")
        s["font_scale"] = round(float(self.font.get()), 2)
        s["evidence_policy"] = self.evidence.get()
        s["auto_activate_trained_model"] = bool(self.auto_act.get())
        s["show_simulation_check"] = bool(self.sim_chk.get())
        s["engineer_pin"] = self.pin.get().strip()
        D.save_settings(s)
        self.saved.configure(text="Saved.")
        self.app.after(2500, lambda: self.saved.configure(text=""))

    def reset(self):
        if not messagebox.askyesno("Reset", "Put every setting back to its default?"):
            return
        self.app.settings = dict(D.DEFAULT_SETTINGS)
        D.save_settings(self.app.settings)
        self.font.set(1.0)
        self.set_font(1.0)
        for key, widget in (("camera_probe", self.probe), ("bench_seconds", self.bsecs),
                            ("monitor_hz", self.hz), ("per_page", self.per_page)):
            widget.delete(0, "end")
            widget.insert(0, str(D.DEFAULT_SETTINGS[key]))
        self.saved.configure(text="Reset.")


def _selftest_label(app):
    """The Label workflow end to end -- keys, inspector, trash + undo, AI accept, folder import -- on a
    throwaway project in a temp folder. The real project's files, labels.csv and active.txt are untouched."""
    import tempfile

    class _Key:
        def __init__(self, keysym, state=0):
            self.keysym, self.state, self.widget = keysym, state, app

    def drain(sec=20.0):
        t = time.monotonic()
        while time.monotonic() - t < sec:
            app.update()
            if lt._pending == 0 and app.q.empty():
                return
            time.sleep(0.02)
        raise AssertionError("label edits never finished")

    lt = app.tab_label
    real = (D.PROJECTS, D.ACTIVE_TXT, D.PROJECT)
    blank = np.full((60, 30, 3), 90, np.uint8)
    td = tempfile.mkdtemp()
    try:
        D.PROJECTS = Path(td) / "projects"
        D.ACTIVE_TXT = D.PROJECTS / "active.txt"
        name = D.create_project("Selftest product")
        for rel in ("+ve/g1.jpg", "+ve/g2.jpg", "-ve/Dent/d1.jpg", "-ve/Scratch/s1.jpg", "_inbox/n1.jpg",
                    "_inbox/n2.jpg"):
            f = D.PROJECTS / name / "images" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", blank)[1].tofile(str(f))
        app.open_project(name)
        app.tabs.set("Label")
        lt.filter_to("All images")
        app.update()
        assert app.defects == ["dent", "scratch"] and len(lt.items) == 6, (app.defects, lt.items)
        assert sum(len(r.winfo_children()) for r in lt.strip.winfo_children()) >= 6, "balance strip not drawn"

        # 1-9 TOGGLE: the first press sets, the second clears (the old tab could only ever set)
        g1 = "+ve/g1.jpg"
        lt.sel = {g1}
        lt.on_key(_Key("1"))
        drain()
        assert app.labels[g1]["dent"] == 1, "key 1 did not set the defect"
        lt.sel = {g1}
        lt.on_key(_Key("1"))
        drain()
        assert app.labels[g1]["dent"] == 0, "key 1 did not toggle the defect off"
        assert [r["action"] for r in D.read_log()][-2:] == ["set dent", "clear dent"], D.read_log()

        # a filter change drops the selection; an off-page selection is reported
        lt.sel = {g1}
        lt.filter_to("Inbox - not reviewed")
        assert not lt.sel, "selection survived a filter change"
        lt.sel = {g1}
        lt.update_sel()
        assert "not on this page" in lt.selinfo.cget("text")
        lt.clear_sel()

        # inspector: review the inbox, label by key, auto-advance to the next image
        lt.inspect(lt.items[0])
        assert lt.inspecting and lt.current == "_inbox/n1.jpg" and lt.sel == {"_inbox/n1.jpg"}
        lt.on_key(_Key("2"))
        drain()
        assert app.labels["_inbox/n1.jpg"]["scratch"] == 1 and app.labels["_inbox/n1.jpg"]["reviewed"] == 1
        assert lt.i_checks["scratch"].get() == 1, "inspector checkbox does not follow the label"
        lt.on_key(_Key("g"))                          # GOOD + auto-advance
        drain()
        assert app.labels["_inbox/n1.jpg"]["scratch"] == 0
        lt.step(1)
        lt.step(-1)
        lt.toggle_inspector()
        assert not lt.inspecting

        # AI suggestion: shown, never training data until accepted
        D.save_suggestions({"model": "fake", "time": "t", "items": {"_inbox/n2.jpg": {"dent": 0.99, "scratch": 0.01}}})
        lt.refresh()
        lt.filter_to("AI suggested - not reviewed")
        assert lt.items == ["_inbox/n2.jpg"], lt.items
        assert app.labels["_inbox/n2.jpg"]["reviewed"] == 0
        assert lt.suggestion("_inbox/n2.jpg") == (["dent"], 0.99, True)
        lt.sel = {"_inbox/n2.jpg"}
        lt.on_key(_Key("Return"))
        drain()
        row = app.labels["_inbox/n2.jpg"]
        assert row["dent"] == 1 and row["scratch"] == 0 and row["reviewed"] == 1, row
        assert "_inbox/n2.jpg" not in D.load_suggestions()["items"]

        # delete goes to the trash; Ctrl+Z brings back the file AND its labels
        lt.filter_to("All images")
        lt.sel = {"-ve/Dent/d1.jpg"}
        lt.submit(lambda: D.delete_images(["-ve/Dent/d1.jpg"]), ["-ve/Dent/d1.jpg"])  # delete() minus the dialog
        drain()
        assert "-ve/Dent/d1.jpg" not in app.labels and len(lt.items) == 5
        assert lt.btn_undo.cget("state") == "normal"
        lt.on_key(_Key("z", state=0x0004))
        drain()
        assert app.labels["-ve/Dent/d1.jpg"]["dent"] == 1 and len(lt.items) == 6, "undo did not restore"

        # folder import into the current product: a new class appears as a column
        src = Path(td) / "incoming"
        for rel in ("Crack/c1.jpg", "Crack/c2.jpg", "ok/o1.jpg"):
            f = src / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", blank)[1].tofile(str(f))
        dlg = ImportDialog(app, src=str(src), modal=False)
        assert dlg.mapping() == {"Crack": "crack", "ok": "GOOD"}, dlg.mapping()
        dlg.where.set("cur")
        dlg.run()
        t = time.monotonic()
        while dlg.win is not None and time.monotonic() - t < 20:
            app.update()
            time.sleep(0.02)
        assert "crack" in app.defects and app.counts["crack"] == 2, app.counts
        assert lt.bdefect.cget("values") and "crack" in lt.bdefect.cget("values")
        assert (src / "Crack/c1.jpg").exists(), "import moved the source files"

        # no model: AI pre-label refuses instead of failing
        app.cfg.pop("active_model", None)
        assert lt.btn_ai.cget("state") == "normal"
    finally:
        D.PROJECTS, D.ACTIVE_TXT = real[0], real[1]
        if lt.inspecting:
            lt.toggle_inspector()
        app.open_project(real[2])
        shutil.rmtree(td, ignore_errors=True)
    assert D.PROJECT == real[2] and D.ACTIVE_TXT.read_text(encoding="utf-8").strip() == real[2]


def A_ok(data) -> bool:
    import annotate
    annotate.validate_annotations(data)
    return True


def selftest():
    """Build every tab, pump the event loop, exercise the grid, then quit.

    A GUI that imports cleanly can still die on the first widget call. This
    constructs the real window against the real dataset, so a bad geometry call
    or a missing attribute fails here instead of in front of the user.
    """
    import tempfile
    app = App()
    # production history of the self-test goes to a temporary folder, never into the real project's record
    _prod_tmp = tempfile.mkdtemp(prefix="selftest_production_")
    app.production_dir = _prod_tmp
    applog.setup(Path(_prod_tmp) / "logs")                    # ...and its log lines too
    mode_before = app.ui_mode
    for _ in range(3):
        app.update()
    for tab in App.TABS:
        app.tabs.set(tab)
        for _ in range(4):
            app.update()

    lt = app.tab_label
    # A brand new install has one empty project, and every tab still has to
    # build. Only assert on the grid when there is something to put in it.
    if app.labels:
        assert lt.items, "label grid loaded no images"
        lt.select_page()
        assert lt.sel, "select page selected nothing"
        lt.clear_sel()
        assert not lt.sel
    for name in lt.MODES:
        lt.mode.set(name)
        lt.goto(0)
        app.update()

    assert D.PROJECT, "no project bound"
    assert app.project.get() == D.project_title(D.PROJECT)
    assert app.tab_defects.upload and app.tab_label.delete   # wired, not just drawn
    assert app.tabs.get() == App.TABS[-1] and set(app.tabs.pages) == set(App.TABS)
    app.update_lamps()
    assert app.lamps["PLC"].val.cget("text") and app.lamps["MODEL"].val.cget("text")
    # operator mode: only the production pages in the rail; engineer mode: everything. Pages are never destroyed.
    app.set_mode("operator", save=False)
    for _ in range(6):                                          # let Tk map / unmap the rail buttons
        app.update(); time.sleep(0.05)
    assert app.tabs.get() in App.OPERATOR_PAGES
    assert app.tabs.buttons["Production"].winfo_ismapped() and not app.tabs.buttons["Label"].winfo_ismapped()
    assert not app.tab_production.eng.winfo_ismapped(), "engineer controls shown to the operator"
    app.set_mode("engineer", save=False)
    for _ in range(6):
        app.update(); time.sleep(0.05)
    assert app.tabs.buttons["Label"].winfo_ismapped() and app.tab_production.eng.winfo_ismapped()
    _selftest_label(app)

    # Analysis must survive both having a model and having none, and must draw
    # curves for a checkpoint saved before curves were ever recorded.
    an = app.tab_analysis
    an.refresh()
    app.update()
    if an.metrics:
        assert an.defect.get() != "-", "a scored model showed no selectable defect"
        an.draw_defect()
        an.metrics = dict(an.metrics, history=[])       # an older checkpoint
        an.draw()
        app.update()
    an.metrics = None
    an.draw()

    # Font scaling is one knob and must survive being pushed to both ends -- with a scrollable page ON SCREEN
    # (the Label grid): that combination once recursed CTkScrollbar.set <-> update_idletasks until the window froze.
    before = float(app.settings.get("font_scale", 1.0))
    app.tabs.set("Label")
    app.update()
    t_font = time.monotonic()
    for v in (0.8, 1.6, before):
        app.tab_settings.set_font(v)
        app.update()
        assert abs(float(app.settings["font_scale"]) - v) < 1e-6, "text size was not saved"
    assert time.monotonic() - t_font < 5, "changing the text size must not rescale the live window"
    assert app.tab_settings.restart_btn.cget("state") == "disabled"      # back on the size in use
    assert abs(float(app.settings["font_scale"]) - before) < 1e-6

    # Multi-camera panes: build for two sources without opening any device.
    app.tab_live.build_panes(["Camera 0", "Camera 1"])
    app.update()
    assert len(app.tab_live.panes) == 2, app.tab_live.panes
    # Detector wiring: default mode is the unchanged classifier path, and missing/foreign weights
    # are refused (raised, not swallowed) -- never silently downgraded to "no detector".
    assert app.tab_live.infer_mode.get() == "Classifier + YOLO" and "YOLO only" in app.tab_live.MODES
    app.settings["detector_weights"] = "no/such/stage2_best.pt"
    try:
        app.tab_live.build_detector()
        raise AssertionError("missing detector weights were not refused")
    except detect.DetectorError:
        pass
    finally:
        app.settings.pop("detector_weights", None)
    # No camera is armed here: the verdict line must say FAULT, never PASS or blank.
    app.tab_live.running = True
    app.tab_live.tick()
    shown = app.tab_live.verdict.cget("text")
    assert shown.startswith("FAULT") and "PASS" not in shown, shown
    app.tab_live.stop()
    app.update()
    assert not app.tab_live.panes

    # Benchmark table must render rows it did not measure, including a failure.
    app.tab_bench.rows = [
        {"requested": "1920x1080@30", "actual_size": "1920x1080", "size_ok": True,
         "fps": 29.0, "latency_ms": 12.0, "jitter_ms": 2.0, "sharpness": 120.0,
         "brightness": 90.0, "clipped": 0.1, "infer_ms": 9.8},
        {"requested": "640x480@30", "error": "cannot open camera"}]
    app.tab_bench.show()
    app.update()
    tb = app.tab_bench                                  # camera lock fields -> camera_controls (no device)
    for k, v in (("focus", "30"), ("exposure", ""), ("wb_temperature", "6500")):
        tb.ctl[k].delete(0, "end"); tb.ctl[k].insert(0, v)
    tb.rot.set("90")
    assert tb._controls_from_ui() == {"focus": 30.0, "autofocus": 0, "wb_temperature": 6500.0,
                                      "auto_wb": 0, "rotate": 90}, tb._controls_from_ui()
    tb.ctl["focus"].delete(0, "end"); tb.ctl["focus"].insert(0, "near")
    try:
        tb._controls_from_ui()
        raise AssertionError("a non-numeric focus was accepted")
    except ValueError:
        pass
    tb.load_controls()

    # Machine tab + conveyor HMI against a FAKE PLC (the self-test never touches a device)
    from plc.test_simulation import FakeLadder, FakePLC, _client
    fake = FakePLC()
    lad = FakeLadder(fake, scale=0.5)
    # generous timings: the camera probes started by other tabs can stall the process for a second or two
    app.plc = PLCService(_client(fake, timeout=1.0), poll_s=0.02, status_period_s=0.1, stale_after_s=6.0)
    mt = app.tab_machine
    app.tabs.set("Machine")
    mt.connect(apply_ui=False)

    def until(cond, sec=12.0):
        t0 = time.time()
        while time.time() - t0 < sec and not cond():
            app.update(); time.sleep(0.02)
        if not cond():
            import faulthandler
            faulthandler.dump_traceback(all_threads=True)
        return cond()
    assert until(lambda: mt.state.cget("text") == "CONNECTED"), (mt.state.cget("text"), mt.msg, app.plc.health_check(),
                                                                   [t.name for t in threading.enumerate()])
    assert set(mt.sw) == {"X0", "X1", "X2", "M0", "M1", "M2"} and "Y0" not in mt.sw and "Y1" not in mt.sw
    mt.open_hmi()
    hmi = mt.hmi
    assert until(lambda: str(mt.sw["M2"][0].cget("state")) == "normal"), "simulator switches not enabled on loopback"
    mt.sw["M2"][1].set(True); mt.sim_input("M2", True)              # the M2 test switch raises a trigger
    assert until(lambda: "TRIGGER RECEIVED" in mt.flow_lbl.cget("text")), mt.flow_lbl.cget("text")
    assert until(lambda: mt.cells["M2"].cget("text") == "ON"), "M2 cell never showed ON"
    assert "AI result: none" in mt.flow_lbl.cget("text")
    assert until(lambda: hmi.bottle is not None and hmi.bottle["mode"] == "station"), "HMI shows no bottle at the station"
    mt.send("REJECT")
    app.update()
    assert not [w for w in fake.writes if w[0] == PLC_AM.address_of("M1")], "command sent without arming"
    mt.arm.select()
    assert until(lambda: str(mt.btn_rej.cget("state")) == "normal")
    mt.send("REJECT")
    assert until(lambda: "ACKED" in mt.cmd_lbl.cget("text")), mt.cmd_lbl.cget("text")
    assert [w for w in fake.writes if w[0] == PLC_AM.address_of("M1")] == [(PLC_AM.address_of("M1"), 1)], fake.writes
    assert until(lambda: hmi.bottle is not None and hmi.bottle["mode"] == "pushed", 4.0), "HMI did not show the reject push"
    assert until(lambda: hmi.bottle is None, 4.0), "rejected bottle never left the HMI"
    assert until(lambda: "PLC ACK" in mt.log.get("1.0", "end"))
    txt = mt.log.get("1.0", "end")
    assert "M2 TRIGGER" in txt and "RESULT" in txt and "Y0 REJECT" in txt, txt
    fake.regs[PLC_AM.address_of("C0")] = 7; fake.regs[PLC_AM.address_of("C1")] = 3     # counters shown as values
    assert until(lambda: mt.cells["C0"].cget("text") == "7" and mt.cells["C1"].cget("text") == "3"), "counters not shown"
    assert until(lambda: "PASS 7  REJECT 3" in mt.flow_lbl.cget("text")), mt.flow_lbl.cget("text")
    assert until(lambda: "C0 PASS COUNT" in mt.log.get("1.0", "end")), "counter change not logged"
    assert not any(w[0] in (PLC_AM.address_of("Y0"), PLC_AM.address_of("Y1")) for w in fake.writes), "a Y output was written"
    hmi.close()
    # live feedback: STOP greys the controls, a dead link shows NO DATA, auto-reconnect restores it without writing
    assert until(lambda: mt.live_lbl.cget("text").startswith("LIVE")), mt.live_lbl.cget("text")
    fake.bits[PLC_AM.address_of("M1000")] = 0
    assert until(lambda: mt.run_lbl.cget("text") == "PLC STOP" and "STOP" in mt.flow_lbl.cget("text")), mt.run_lbl.cget("text")
    assert until(lambda: str(mt.sw["X0"][0].cget("state")) == "disabled"), "test inputs stayed enabled in STOP"
    fake.bits[PLC_AM.address_of("M1000")] = 1
    assert until(lambda: mt.run_lbl.cget("text") == "PLC RUN")
    n_w = len(fake.writes)
    fake.mode = "close"
    assert until(lambda: mt.state.cget("text") == "FAULT" and mt.live_lbl.cget("text").startswith("NO DATA")), mt.state.cget("text")
    assert mt.cells["M2"].cget("text") == "--", "stale value shown on a dead link"
    fake.mode = "ok"
    assert until(lambda: mt.state.cget("text") == "CONNECTED", 15.0), "auto-reconnect did not restore the link"
    assert len(fake.writes) == n_w, "auto-reconnect wrote to the PLC"
    mt.disconnect()
    assert until(lambda: mt.state.cget("text") == "DISCONNECTED")
    t_end = time.time() + 4.0
    while time.time() < t_end:
        app.update(); time.sleep(0.02)
    assert mt.state.cget("text") == "DISCONNECTED", "auto-reconnect ignored the operator's Disconnect"
    mt.connect(apply_ui=False)
    assert until(lambda: mt.state.cget("text") == "CONNECTED")
    # link panel: both modes build a transport from the fields without opening anything
    mt.mode.set("Simulator (TCP)"); mt._mode_changed()
    assert mt.link_from_ui()["plc_mode"] == "tcp" and plc_link(mt.link_from_ui())[0].description.startswith("TCP ")
    mt.mode.set("Real PLC (serial)"); mt._mode_changed(); app.update()
    mt._com_map = {"COM99 - test": "COM99"}; mt.com.configure(values=["COM99 - test"]); mt.com.set("COM99 - test")
    lk = plc_link(mt.link_from_ui())
    assert lk[0].description == "SERIAL COM99 9600 7E1" and "NOT yet verified" in mt.conn_note.cget("text"), lk[0].description
    mt.station.delete(0, "end"); mt.station.insert(0, "x")
    try:
        mt.link_from_ui(); raise AssertionError("bad station accepted")
    except ValueError:
        pass
    mt.station.delete(0, "end"); mt.station.insert(0, "1")
    mt.mode.set("Simulator (TCP)"); mt._mode_changed()
    mt.test_link()
    assert until(lambda: "link test: 10/10" in mt.info.cget("text")), mt.info.cget("text")

    # Production tab: the whole inspection line against the same FAKE PLC (decoded-ladder emulation) with
    # fake cameras and a fake detector -- no device, no weights. Start line saves settings: restore them.
    pt = app.tab_production
    app.tabs.set("Production")
    settings_before = D.SETTINGS_JSON.read_text() if D.SETTINGS_JSON.exists() else None
    scene = {"kind": "good"}

    class _Cap:
        def isOpened(self):
            return True
        def get(self, *_):
            return 0
        def set(self, *_):
            return True
        def release(self):
            pass
        def read(self):
            time.sleep(0.02)
            img = np.zeros((300, 200, 3), np.uint8)
            img[0, 0, 0] = 2 if scene["kind"] == "nocap" else 1
            return True, img

    def _yolo(frame):
        boxes = [([60, 40, 140, 290], 0.95, 0), ([80, 25, 120, 60], 0.9, 1), ([65, 120, 135, 220], 0.88, 2)]
        if int(frame[0, 0, 0]) == 2:
            boxes.pop(1)                                             # no cap
        return [b[0] for b in boxes], [b[1] for b in boxes], [b[2] for b in boxes]

    real_open = infer.open_capture
    infer.open_capture = lambda src, *_: _Cap()
    try:
        # the operator's saved line_cameras (real indices, e.g. [2, 3]) must not decide what the fakes get
        app.settings.pop("line_cameras", None)
        pt._set_cams([{"index": 0, "width": 640, "height": 480, "name": "EMEET SmartCam Nova 4K"},
                      {"index": 1, "width": 640, "height": 480, "name": "EMEET SmartCam Nova 4K"}])
        assert pt.chosen() == [0, 1], pt.chosen()
        pt.task.set("Detection")
        for key, val in (("plc_t0_s", "0.75"), ("plc_t1_s", "0.25"), ("inspect_frames", "2"),
                         ("inspection_to_reject_mm", "0"), ("conveyor_mm_s", "0")):
            pt.tim[key].delete(0, "end"); pt.tim[key].insert(0, val)
        pt.tim["plc_t0_s"].delete(0, "end"); pt.tim["plc_t0_s"].insert(0, "x")
        assert not pt.save_timing() and "not a number" in pt.tim_msg.cget("text")
        pt.tim["plc_t0_s"].delete(0, "end"); pt.tim["plc_t0_s"].insert(0, "15")    # the K150 ladder
        assert not pt.save_timing() and "will not start" in pt.tim_msg.cget("text")
        pt.tim["plc_t0_s"].delete(0, "end"); pt.tim["plc_t0_s"].insert(0, "0.75")
        assert until(lambda: pt.mstate == MS.READY), (pt.mstate, pt.mreason)        # all start checks pass
        pt.start_line(models={"detection": detect.YoloDetector(model=detect.FakeYolo(_yolo), warmup=False)})
        assert until(lambda: pt.line_lbl.cget("text") in ("RUNNING", "INSPECTING")), pt.line_lbl.cget("text")
        assert until(lambda: pt.cam_lbl["Camera 0"].cget("text").startswith("CONNECTED")
                     and pt.cam_lbl["Camera 1"].cget("text").startswith("CONNECTED")), pt.cam_lbl["Camera 0"].cget("text")
        assert "200x300" in pt.cam_lbl["Camera 0"].cget("text")
        for kind, want in (("good", infer.PASS), ("nocap", infer.REJECT), ("good", infer.PASS)):
            scene["kind"] = kind
            n0 = pt.line.counts["total"]
            lad.trigger(hold_s=0.05)
            assert until(lambda: pt.line.counts["total"] > n0, 10.0), (kind, pt.line.snapshot()["alarms"])
            last = pt.line.snapshot()["history"][-1]
            assert last.final == want, (kind, last.final, last.reason, last.plc_status)
        assert until(lambda: pt.cnt["total"].cget("text") == "3" and pt.cnt[infer.REJECT].cget("text") == "1")
        table = pt.table.get("1.0", "end")
        assert "000001" in table and "000003" in table and "missing_cap" in table, table
        assert pt.res_lbl.cget("text") == "GOOD" and "000003" in pt.ins_lbl.cget("text")
        assert "Y0 pulse" in table, table
        assert app.plc.watch_x0 and app.plc.untriggered == 0
        # a line camera that dies while running is reopened in the background (no app restart)
        class _DeadCam:
            error, alive, frame_wh, fps = "camera stopped returning frames", False, None, 0.0
            def __init__(self):
                self.calls = []
            def stop(self):
                self.calls.append("stop")
            def start(self, src):
                self.calls.append(("start", src))
        dead = _DeadCam()
        pt._camera_watch("Camera 9", dead, 7)
        assert until(lambda: ("start", 7) in dead.calls, 3.0) and pt._reopens["Camera 9"] == 1, dead.calls
        pt._camera_watch("Camera 9", dead, 7)                     # within RECONNECT_S: not hammered
        assert pt._reopens["Camera 9"] == 1
        assert any("Camera 9 (#7) reconnect attempt 1" in x for x in applog.search("camera", "Camera 9"))
        # the recipe editor validates against the detector classes and refuses to save while the line runs
        import hmi as hmi_mod                                      # "hmi" is the conveyor HMI window here
        rd = hmi_mod.RecipeDialog(app, detect.CLASS_NAMES)
        app.update()
        assert rd.check(), rd.msg.cget("text")
        rd.rows[0]["name"].delete(0, "end"); rd.rows[0]["name"].insert(0, "lid")
        assert not rd.check() and "no class lid" in rd.msg.cget("text")
        assert not rd.save() and "Stop the line" in rd.msg.cget("text")
        rd.destroy()
        sd = hmi_mod.SimulationCheckDialog(app)
        app.update()
        sd.check_ladder()
        assert "SOFTWARE" in sd.verdict.cget("text").upper() or "LADDER" in sd.verdict.cget("text").upper(), sd.verdict.cget("text")
        assert "S4" in sd.out.get("1.0", "end") and "FAIL" in sd.out.get("1.0", "end") or "PASS" in sd.out.get("1.0", "end")
        sd.destroy()
        assert app.lamps["LINE"].val.cget("text") in ("RUNNING", "INSPECTING")       # one state, everywhere
        pt.halt_line()                                             # software HALT: latched until RESET FAULT
        assert until(lambda: pt.line_lbl.cget("text") == "FAULT"), pt.line_lbl.cget("text")
        assert any(a.code == "LINE_HALTED" for a in app.alarms.active())
        n_w = len(fake.writes)
        scene["kind"] = "good"
        n0 = pt.line.counts["total"]
        lad.trigger(hold_s=0.05)
        assert until(lambda: pt.line.counts["total"] > n0, 10.0)
        assert len(fake.writes) == n_w, "the halted line wrote to the PLC"
        pt.reset_halt()
        assert until(lambda: pt.line_lbl.cget("text") in ("RUNNING", "INSPECTING"))
        assert not any(a.code == "LINE_HALTED" for a in app.alarms.active()), "RESET FAULT did not acknowledge"
        pt.stop_line()
        assert until(lambda: pt.line_lbl.cget("text") in ("READY", "NOT READY")) and not app.cams.running()
        assert not app.plc.watch_x0
        # every bottle is in the persistent record (temporary folder), with the run's model versions
        st = app.store()
        st.flush()
        rows = st.recent(20)
        assert len(rows) == pt.line.counts["total"] == 4 and str(st.folder) == _prod_tmp, (len(rows), st.folder)
        assert {r["final"] for r in rows} == {infer.PASS, infer.REJECT, infer.FAULT}
        assert any(r["evidence"] for r in rows if r["final"] == infer.REJECT)
        app.tab_history.refresh()
        assert app.tab_history.list.size() == 4 and app.tab_history.cnt["REJECT"].cget("text") == "1"
        shift_now = next(n for n, a, b in app.tab_history.shifts()
                         if production_store.shift_span(time.strftime("%Y-%m-%d"), (n, a, b))[0] <= time.time()
                         < production_store.shift_span(time.strftime("%Y-%m-%d"), (n, a, b))[1])             if any(production_store.shift_span(time.strftime("%Y-%m-%d"), x)[0] <= time.time()
                   < production_store.shift_span(time.strftime("%Y-%m-%d"), x)[1] for x in app.tab_history.shifts()) else None
        if shift_now:                                              # the current shift holds today's 4 bottles
            app.tab_history.shift.set(shift_now)
            app.tab_history.refresh()
            assert app.tab_history.cnt["total"].cget("text") == "4", app.tab_history.cnt["total"].cget("text")
            app.tab_history.shift.set("Whole day")
        app.tabs.set("Database")
        app.tab_database.refresh()
        assert len(app.tab_database.rows) == 4 and app.tab_database.cols[2] == "result", app.tab_database.cols
        assert {r[2] for r in app.tab_database.rows} == {"GOOD", "DEFECT", "FAULT"}
        app.tab_database.table.set("Alarms"); app.tab_database.refresh()
        app.tab_database.table.set("Runs"); app.tab_database.refresh()
        assert len(app.tab_database.rows) == 1
        app.tab_database.table.set("Bottles")
        app.tab_health.update()
        app.tab_health.show_logs()
        assert "line started" in app.tab_health.log_box.get("1.0", "end") or             any("line started" in x for x in applog.search("machine"))
        app.tab_models.refresh()
        assert app.tab_models.list.size() >= 1
        assert not any(w[0] in (PLC_AM.address_of("Y0"), PLC_AM.address_of("Y1")) for w in fake.writes)
    finally:
        pt.stop_line()
        infer.open_capture = real_open
        if settings_before is not None:
            D.SETTINGS_JSON.write_text(settings_before)
        app.settings = D.load_settings()
    app.plc.stop()
    lad.stop()
    fake.close()

    # Annotate tab: model proposals stay out of `boxes` until a person accepts them
    import tempfile
    import autoannotate as AA
    at = app.tab_annotate
    with tempfile.TemporaryDirectory() as td:
        cv2.imwrite(str(Path(td) / "a.jpg"), np.zeros((120, 80, 3), np.uint8))
        at.load_external(Path(td), Path(td) / "ann.json", "detection", ["bottle", "cap"], ["a.jpg"])
        AA.propose(at.data, "a.jpg", [("bottle", 0.9, 5, 5, 70, 110), ("cap", 0.8, 20, 2, 50, 20)], (80, 120))
        at.load_image("a.jpg")
        assert not at.data["images"]["a.jpg"]["boxes"] and len(at.data["images"]["a.jpg"]["proposals"]) == 2
        at.accept_proposals()
        e = at.data["images"]["a.jpg"]
        assert len(e["boxes"]) == 2 and not e["proposals"] and not e["reviewed"], e
        assert A_ok(at.data)
    app.cams.stop()
    app.ui_mode = mode_before
    if app._store is not None:
        app._store.close()
        app._store = None
    app.destroy()
    applog.setup()                                             # release the temp log files before removing them
    shutil.rmtree(_prod_tmp, ignore_errors=True)
    print(f"ok  project {D.PROJECT!r}: {len(app.labels)} images, "
          f"{len(app.defects)} defects, all {len(App.TABS)} tabs built")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        App().mainloop()
