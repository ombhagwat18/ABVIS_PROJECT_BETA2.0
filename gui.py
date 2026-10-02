"""Bottle Inspection - desktop dashboard.

Label images, manage defect types, retrain, and watch the line. All the work
lives in dataset/train/infer/calibrate; this file is only the window.

Tk is not thread-safe: every widget touch happens on the main thread, and
background work (training, thumbnail loading, the camera) reports back through
a queue drained by after().
"""
from __future__ import annotations

import csv
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

import annotation_studio
import bench
import charts
import dataset as D
import infer

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# White and blue. Red, amber and green survive only where they carry a meaning
# a colour scheme has no business overriding: a reject an operator reads across
# a room, a warning, a passing recall. Everything decorative is blue.
ACC, ACC_H, ACC_T = "#1d4ed8", "#1e40af", "#ffffff"
BAD, GOOD, WARN, DIM = "#b91c1c", "#15803d", "#b45309", "#5b6672"
PANEL, LINE, BG, INK = "#f1f5fb", "#d8e2f0", "#ffffff", "#0f172a"
VIDEO_BG = "#0f172a"          # video needs a dark backing whatever the theme
PER_PAGE = 60
MONO = ("Consolas", 12)


def bgr_to_ctk(frame: np.ndarray, size=None) -> ctk.CTkImage:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if size:
        img.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)


def chart_panel(parent, title, w, h, note) -> ctk.CTkCanvas:
    """A titled chart canvas + caption, the chrome shared by every chart in
    AnalysisTab and TrainTab."""
    box = ctk.CTkFrame(parent, fg_color=PANEL)
    box.pack(side="left", fill="both", expand=True, padx=(0, 8))
    ctk.CTkLabel(box, text=title, text_color=DIM,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
    cv = ctk.CTkCanvas(box, width=w, height=h, bg=BG, highlightthickness=0)
    cv.pack(fill="both", expand=True, padx=12, pady=(0, 4))
    ctk.CTkLabel(box, text=note, text_color=DIM, font=("Segoe UI", 10),
                 wraplength=w - 10, justify="left").pack(anchor="w", padx=12, pady=(0, 10))
    return cv


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Bottle Inspection")
        self.geometry("1500x950")
        self.minsize(1150, 700)

        self.defects: list[str] = []
        self.labels: dict = {}
        self.counts: dict = {}
        self.cfg: dict = {}
        self.cams = infer.CameraSet()
        self.q: queue.Queue = queue.Queue()

        self.settings = D.load_settings()
        self.apply_font_scale(self.settings.get("font_scale", 1.0), save=False)

        head = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        head.pack(fill="x")
        ctk.CTkLabel(head, text="INSPECT", font=("Segoe UI", 15, "bold"),
                     text_color=ACC).pack(side="left", padx=(16, 12), pady=10)
        self._pmap: dict[str, str] = {}
        self.project = ctk.CTkOptionMenu(head, values=["-"], width=210,
                                         command=self.switch_project)
        self.project.pack(side="left", padx=(0, 6))
        ctk.CTkButton(head, text="+ New project", width=110, fg_color="transparent",
                      border_width=1, command=self.new_project).pack(side="left")
        self.status = ctk.CTkLabel(head, text="", text_color=DIM, font=("Segoe UI", 12))
        self.status.pack(side="right", padx=16)

        self.tabs = ctk.CTkTabview(self, fg_color="transparent")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        for name in self.TABS:
            self.tabs.add(name)

        self.tab_label = LabelTab(self, self.tabs.tab("Label"))
        self.tab_defects = DefectsTab(self, self.tabs.tab("Defects"))
        self.tab_train = TrainTab(self, self.tabs.tab("Train"))
        self.tab_analysis = AnalysisTab(self, self.tabs.tab("Analysis"))
        self.tab_live = LiveTab(self, self.tabs.tab("Live"))
        self.tab_bench = BenchTab(self, self.tabs.tab("Camera"))
        self.tab_data = DataTab(self, self.tabs.tab("Data health"))
        self.tab_annotate = annotation_studio.AnnotationTab(self, self.tabs.tab("Annotate"))
        self.tab_settings = SettingsTab(self, self.tabs.tab("Settings"))

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.reload()
        self.after(60, self.pump)

    TABS = ("Label", "Defects", "Train", "Analysis", "Live", "Camera",
            "Data health", "Annotate", "Settings")

    def all_tabs(self):
        return (self.tab_label, self.tab_defects, self.tab_train, self.tab_analysis,
                self.tab_live, self.tab_bench, self.tab_data, self.tab_annotate,
                self.tab_settings)

    def apply_font_scale(self, scale: float, save: bool = True):
        """One knob for text size: CustomTkinter scales fonts with the widgets,
        so scaling both keeps padding proportional instead of leaving big text
        clipped inside buttons sized for small text."""
        scale = max(0.7, min(1.8, float(scale)))
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
        self.after(60, self.pump)

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
        # that no longer belongs to the data on screen.
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

        ctk.CTkLabel(win, text="Name for the new project", font=("Segoe UI", 13)).pack(
            padx=20, pady=(20, 6), anchor="w")
        name_entry = ctk.CTkEntry(win, width=320)
        name_entry.pack(padx=20)
        name_entry.focus()

        ctk.CTkLabel(win, text="Task", font=("Segoe UI", 13)).pack(
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
        self.cams.stop()
        D.use_project(name)
        self.reload()
        if task == "classification":
            hint = (f"{title!r} is empty.\n\nGo to Defect types: 'Upload GOOD images' for bottles "
                    f"that pass, then add a defect type and use its Upload button for the ones "
                    f"that fail.")
        else:
            hint = (f"{title!r} is empty and set up for {task}.\n\nAdd images under its images/ "
                    f"folder, then use the Annotate tab to label them.")
        messagebox.showinfo("Project created", hint)

    def reload(self):
        names = D.list_projects()
        self._pmap = {D.project_title(n): n for n in names}
        self.project.configure(values=list(self._pmap) or ["-"])
        self.project.set(D.project_title(D.PROJECT))

        self.defects, self.labels = D.load_labels()
        self.counts = D.counts(self.defects, self.labels)
        self.cfg = D.load_config()
        c = self.counts
        model = self.cfg.get("active_model") or "no model"
        self.status.configure(
            text=f"{c['_total']} images   ·   {c['_good']} good   ·   "
                 f"{c['_unreviewed']} unreviewed   ·   {model}")
        for t in self.all_tabs():
            t.refresh()

    def on_close(self):
        self.cams.stop()
        self.destroy()


# --------------------------------------------------------------------- Label
class LabelTab:
    """Thumbnail grid, multi-select, bulk labelling, keyboard shortcuts."""

    # No "+ve/-ve" here either: at dataset level it means pass/fail, at defect
    # level it means has/has-not, and one label for both reads as the wrong one.
    MODES = {"All images": "all",
             "Inbox - not reviewed": "inbox",
             "GOOD - passes": "good",
             "DEFECTIVE - any defect": "defective",
             "WITH this defect": "pos",
             "WITHOUT this defect": "neg"}

    def __init__(self, app: App, parent):
        self.app, self.page, self.sel, self.items = app, 0, set(), []
        self.cards: dict[str, ctk.CTkFrame] = {}
        self._imgs: list = []          # Tk drops images that nothing references
        self._token = 0                # ignore thumbnails from a superseded page
        self.last_click = None         # anchor for shift-click range select
        self._search_after = None      # debounce handle for live search

        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.pack(fill="x", pady=(0, 8))
        self.mode = ctk.CTkOptionMenu(bar, values=list(self.MODES), width=200,
                                      command=lambda _: self.goto(0))
        self.mode.pack(side="left", padx=8, pady=8)
        self.defect = ctk.CTkOptionMenu(bar, values=["-"], width=170,
                                        command=lambda _: self.goto(0))
        self.defect.pack(side="left", padx=(0, 8))
        self.search = ctk.CTkEntry(bar, placeholder_text="filename filter", width=160)
        self.search.pack(side="left", padx=(0, 8))
        self.search.bind("<Return>", lambda e: self.goto(0))
        self.search.bind("<KeyRelease>", self.on_search_key)
        self.info = ctk.CTkLabel(bar, text="", text_color=DIM)
        self.info.pack(side="left", padx=8)
        ctk.CTkButton(bar, text="→", width=42, command=lambda: self.goto(self.page + 1)).pack(side="right", padx=(0, 8))
        ctk.CTkButton(bar, text="←", width=42, command=lambda: self.goto(self.page - 1)).pack(side="right", padx=4)

        bulk = ctk.CTkFrame(parent, fg_color=PANEL)
        bulk.pack(fill="x", pady=(0, 8))
        self.selinfo = ctk.CTkLabel(bulk, text="0 selected", font=("Segoe UI", 13, "bold"))
        self.selinfo.pack(side="left", padx=10, pady=8)
        ctk.CTkButton(bulk, text="Select page", width=100, command=self.select_page).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Select all", width=90, command=self.select_all).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Clear", width=70, command=self.clear_sel).pack(side="left", padx=3)
        self.bdefect = ctk.CTkOptionMenu(bulk, values=["-"], width=170)
        self.bdefect.pack(side="left", padx=(14, 4))
        ctk.CTkButton(bulk, text="Set defect", width=100, fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=lambda: self.apply(1)).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Clear defect", width=105, command=lambda: self.apply(0)).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Mark GOOD", width=105, command=self.mark_good).pack(side="left", padx=3)
        ctk.CTkButton(bulk, text="Delete", width=75, fg_color="transparent", border_width=1,
                      text_color=BAD, command=self.delete).pack(side="left", padx=(14, 3))
        ctk.CTkLabel(bulk, text="1-9 toggle defect · G good · A select page · Esc clear · ←/→ page · Shift-click range",
                     text_color=DIM, font=("Segoe UI", 11)).pack(side="right", padx=10)

        self.grid = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.grid.pack(fill="both", expand=True)

        app.bind("<Key>", self.on_key)

    # keyboard -------------------------------------------------------------
    def on_key(self, e):
        if self.app.tabs.get() != "Label":
            return
        if isinstance(e.widget, (ctk.CTkEntry,)) or e.widget.winfo_class() == "Entry":
            return
        k = e.keysym.lower()
        if k in "123456789" and k.isdigit():
            i = int(k) - 1
            if i < len(self.app.defects):
                self.bdefect.set(self.app.defects[i])
                self.apply(1)
        elif k == "g":
            self.mark_good()
        elif k == "a":
            self.select_page()
        elif k == "escape":
            self.clear_sel()
        elif k == "left":
            self.goto(self.page - 1)
        elif k == "right":
            self.goto(self.page + 1)

    def on_search_key(self, e):
        if self._search_after is not None:
            self.app.after_cancel(self._search_after)
        self._search_after = self.app.after(250, lambda: self.goto(0))

    # data -----------------------------------------------------------------
    def refresh(self):
        ds = self.app.defects or ["-"]
        for m in (self.defect, self.bdefect):
            cur = m.get()
            m.configure(values=ds)
            m.set(cur if cur in ds else ds[0])
        self.load()

    def query(self):
        mode = self.MODES[self.mode.get()]
        d, q = self.defect.get(), self.search.get().strip().lower()
        out = []
        for p in sorted(self.app.labels, key=D._natkey):
            row = self.app.labels[p]
            if q and q not in p.lower():
                continue
            reviewed = row.get("reviewed", 1)
            if mode == "inbox" and reviewed:
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

    def goto(self, page):
        self.page = max(0, page)
        self.load()

    def load(self):
        all_paths = self.query()
        per = max(6, int(self.app.settings.get("per_page", PER_PAGE)))
        pages = max(1, -(-len(all_paths) // per))
        self.page = min(self.page, pages - 1)
        self.items = all_paths[self.page * per:(self.page + 1) * per]
        self.info.configure(text=f"{len(all_paths)} images   ·   page {self.page + 1}/{pages}")

        for w in self.grid.winfo_children():
            w.destroy()
        self.cards.clear()
        self._imgs.clear()
        self._token += 1
        token = self._token

        cols = 9
        for i, p in enumerate(self.items):
            card = ctk.CTkFrame(self.grid, fg_color=PANEL, border_width=2,
                                border_color=ACC if p in self.sel else PANEL)
            card.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="nsew")
            ph = ctk.CTkLabel(card, text="", width=132, height=132)
            ph.pack(padx=4, pady=(4, 0))
            row = self.app.labels[p]
            tags = [d for d in self.app.defects if row.get(d)]
            txt = ", ".join(tags) if tags else ("good" if row.get("reviewed", 1) else "new")
            col = BAD if tags else (GOOD if row.get("reviewed", 1) else WARN)
            ctk.CTkLabel(card, text=txt[:22], text_color=col, font=("Segoe UI", 10),
                         wraplength=130).pack(padx=4, pady=(2, 5))
            for w in (card, ph):
                w.bind("<Button-1>", lambda e, p=p: self.click(p, e))
                w.bind("<Double-Button-1>", lambda e, p=p: self.open_full(p))
            self.cards[p] = (card, ph)

        # Decoding 60 JPEGs blocks the window for a beat; do it off-thread and
        # drop the results if the user has already paged away.
        paths = list(self.items)

        def work():
            for p in paths:
                b = D.thumbnail(p)
                if b is None:
                    continue
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                img = bgr_to_ctk(arr, (132, 132))
                self.app.post(lambda p=p, img=img: self.set_thumb(token, p, img))
        threading.Thread(target=work, daemon=True).start()
        self.update_sel()

    def set_thumb(self, token, path, img):
        if token != self._token or path not in self.cards:
            return
        self._imgs.append(img)
        self.cards[path][1].configure(image=img, text="")

    # selection ------------------------------------------------------------
    def toggle(self, p):
        self.sel.symmetric_difference_update({p})
        self.paint(p)
        self.update_sel()

    def click(self, p, event):
        shift = bool(event.state & 0x0001)
        if shift and self.last_click in self.items and p in self.items:
            lo, hi = sorted((self.items.index(self.last_click), self.items.index(p)))
            rng = self.items[lo:hi + 1]
            self.sel.update(rng)
            for q in rng:
                self.paint(q)
            self.update_sel()
        else:
            self.toggle(p)
        self.last_click = p

    def paint(self, p):
        if p in self.cards:
            self.cards[p][0].configure(border_color=ACC if p in self.sel else PANEL)

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

    def update_sel(self):
        self.selinfo.configure(text=f"{len(self.sel)} selected")

    def open_full(self, p):
        img = D.imread(D.IMAGE_ROOT / p)
        if img is None:
            return
        win = ctk.CTkToplevel(self.app)
        win.title(p)
        win.geometry("1100x800")
        win.after(200, win.lift)
        h, w = img.shape[:2]
        s = min(1050 / w, 700 / h)
        shown = cv2.resize(img, (round(w * s), round(h * s)))
        im = bgr_to_ctk(shown)
        lab = ctk.CTkLabel(win, text="", image=im)
        lab.image = im
        lab.pack(padx=10, pady=10)
        row = self.app.labels.get(p, {})
        tags = [d for d in self.app.defects if row.get(d)]
        ctk.CTkLabel(win, text=", ".join(tags) or "good bottle (no defects)",
                     text_color=BAD if tags else GOOD).pack()

    # edits ----------------------------------------------------------------
    def apply(self, value):
        if not self.sel:
            return messagebox.showinfo("Nothing selected", "Select some images first.")
        d, paths = self.bdefect.get(), list(self.sel)
        self.app.run_bg(lambda: D.apply_labels(paths, d, value), lambda _: self.after_edit())

    def delete(self):
        n = len(self.sel)
        if not n:
            return messagebox.showinfo("Delete", "Select some images first.")
        if not messagebox.askyesno("Delete images",
                                   f"Permanently delete {n} image file(s) from disk?\n"
                                   f"This cannot be undone."):
            return
        paths = list(self.sel)
        self.app.run_bg(lambda: D.delete_images(paths),
                        lambda _: (self.clear_sel(), self.app.reload()))

    def mark_good(self):
        if not self.sel:
            return messagebox.showinfo("Nothing selected", "Select some images first.")
        paths = list(self.sel)
        self.app.run_bg(lambda: D.apply_labels(paths, clear_all=True), lambda _: self.after_edit())

    def after_edit(self):
        self.sel.clear()
        self.app.reload()


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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
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
                         font=("Segoe UI", 11)).pack(anchor="w")
            if title.startswith("GOOD"):
                ctk.CTkButton(box, text="Upload GOOD images", width=170, fg_color=ACC,
                              text_color=ACC_T, hover_color=ACC_H,
                              command=lambda: self.upload(None)).pack(anchor="w", pady=(6, 0))
        share = good / max(1, good + bad)
        ctk.CTkLabel(top, wraplength=820, justify="left", padx=0,
                     text_color=WARN if share < 0.5 else GOOD,
                     text=(f"Only {share:.0%} of the labelled set passes. On a real line good is "
                           f"the overwhelming majority, so the model is seeing a world where "
                           f"defects are normal — grow the good set."
                           if share < 0.5 else
                           f"{share:.0%} of the labelled set passes.")
                     ).pack(anchor="w", padx=14, pady=(0, 12))

        ctk.CTkLabel(self.list, text="PER-DEFECT", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self.list, text_color=DIM, justify="left", wraplength=820,
                     text="How many images carry each defect. A bottle can have several, so these "
                          "add up to more than the defective count above.\n"
                          "WITHOUT is not the same as good: it is every other image, including "
                          f"the {c.get('_defective', 0)} that fail on a different defect."
                     ).pack(anchor="w", pady=(0, 6))

        hdr = ctk.CTkFrame(self.list, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 4))
        for t, wdt in (("DEFECT", 220), ("WITH", 80), ("WITHOUT", 90), ("STATUS", 330)):
            ctk.CTkLabel(hdr, text=t, width=wdt, anchor="w", text_color=DIM,
                         font=("Segoe UI", 11, "bold")).pack(side="left")

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
                     font=("Segoe UI", 11, "bold")).pack(side="left", padx=(10, 10), pady=8)
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

        self.log = ctk.CTkTextbox(left, height=180, font=MONO, fg_color="#0a0c0e",
                                   text_color="#e5e7eb")
        self.log.pack(fill="x")
        self.log.insert("end", "No training run in this session.\n")
        self.metrics = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.metrics.pack(fill="both", expand=True, pady=(8, 0))

        right = ctk.CTkFrame(body, fg_color=PANEL, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="MODEL VERSIONS", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
        self.models = ctk.CTkScrollableFrame(right, fg_color="transparent", height=220)
        self.models.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkLabel(right, text="BENCHMARK", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(0, 2))
        ctk.CTkLabel(right, text_color=DIM, font=("Segoe UI", 10), wraplength=310, justify="left",
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
                         font=("Segoe UI", 12, "bold" if s == active else "normal")).pack(side="left")
            if s == active:
                ctk.CTkLabel(row, text="active", text_color=DIM, font=("Segoe UI", 10)).pack(side="left", padx=6)
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
                             font=("Segoe UI", 10, "bold")).pack(side="left")
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
                         font=("Segoe UI", 11, "bold")).pack(side="left")
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

        ctk.CTkLabel(self.metrics, text_color=DIM, justify="left", wraplength=900,
                     text="Recall is the number that matters - a missed defect ships. MISSED is "
                          "false negatives on validation.").pack(anchor="w", pady=(6, 10))

        bad = m.get("mistakes", [])
        if bad:
            ctk.CTkLabel(self.metrics, text=f"SHOW ME THE MISTAKES  ({len(bad)})",
                         text_color=DIM, font=("Segoe UI", 11, "bold")).pack(anchor="w")
            ctk.CTkLabel(self.metrics, text_color=DIM, justify="left", wraplength=900,
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
                             font=("Segoe UI", 9),
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

    def __init__(self, app: App, parent):
        self.app = app
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
        self.verdict = ctk.CTkLabel(bar, text="", font=("Segoe UI", 17, "bold"))
        self.verdict.pack(side="left", padx=20)
        self.hint = ctk.CTkLabel(bar, text="", text_color=DIM)
        self.hint.pack(side="right", padx=10)

        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.grid = ctk.CTkFrame(left, fg_color=VIDEO_BG, corner_radius=8)
        self.grid.pack(fill="both", expand=True)
        self.idle = ctk.CTkLabel(self.grid, text="Cameras stopped.\nTick a source and press Start.",
                                 text_color="#94a3b8")
        self.idle.pack(expand=True)

        # Real-time performance, §2.6. One row per running camera plus the
        # machine's own numbers -- the answer to "is the pipeline keeping up".
        mon = ctk.CTkFrame(left, fg_color=PANEL)
        mon.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(mon, text="REAL-TIME PERFORMANCE", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(src, text="Tick every camera that watches this bottle.",
                     text_color=DIM, font=("Segoe UI", 11), wraplength=300,
                     justify="left").pack(anchor="w", padx=12)
        self.srcbox = ctk.CTkFrame(src, fg_color="transparent")
        self.srcbox.pack(fill="x", padx=12, pady=(6, 10))

        cap = ctk.CTkFrame(right, fg_color=PANEL)
        cap.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(cap, text="SNAPSHOT INTO A LABEL", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.snapfrom = ctk.CTkOptionMenu(cap, values=["-"], width=300)
        self.snapfrom.pack(padx=12, pady=(0, 6))
        self.capbox = ctk.CTkFrame(cap, fg_color="transparent")
        self.capbox.pack(fill="x", padx=12)
        ctk.CTkButton(cap, text="Capture frame", fg_color=ACC, text_color=ACC_T,
                      hover_color=ACC_H, command=self.capture).pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(cap, text="Tick nothing to capture a good bottle.", text_color=DIM,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(0, 6))

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
                                     font=("Segoe UI", 11), wraplength=150, anchor="w")
        self.shot_txt.pack(anchor="w")
        self.undo_btn = ctk.CTkButton(side, text="Undo - delete it", height=26,
                                      fg_color="transparent", border_width=1, text_color=BAD,
                                      command=self.undo_capture)

        th = ctk.CTkFrame(right, fg_color=PANEL)
        th.pack(fill="both", expand=True)
        ctk.CTkLabel(th, text="THRESHOLDS", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.thbox = ctk.CTkScrollableFrame(th, fg_color="transparent")
        self.thbox.pack(fill="both", expand=True, padx=6)
        ctk.CTkLabel(th, text="Lower = catches more, fails more good bottles.",
                     text_color=DIM, font=("Segoe UI", 11), wraplength=300,
                     justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        self.scanned = False

    def refresh(self):
        for w in self.capbox.winfo_children():
            w.destroy()
        self.checks.clear()
        for d in self.app.defects:
            cb = ctk.CTkCheckBox(self.capbox, text=d, font=("Segoe UI", 12))
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
            ctk.CTkLabel(head, text=d, font=("Segoe UI", 11)).pack(side="left")
            val = ctk.CTkLabel(head, text=f"{float(t.get(d, 0.5)):.2f}", font=("Segoe UI", 11, "bold"))
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
            cb = ctk.CTkCheckBox(self.srcbox, text=name, font=("Segoe UI", 12))
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
        stamp = cfg.get("active_model")
        for name, val in chosen:
            self.app.cams.add(val, name)
        try:
            self.app.cams.start([v for _, v in chosen], stamp)
        except Exception as e:
            messagebox.showwarning("Model", f"Running without a model: {e}")
            self.app.cams.start([v for _, v in chosen], None)
        self.build_panes([n for n, _ in chosen])
        self.running = True
        self.btn_start.configure(state="disabled")
        self.app.after(1200, self.check_started)
        self.tick()

    def build_panes(self, names):
        for w in self.grid.winfo_children():
            w.destroy()
        self.panes.clear()
        self._imgs.clear()
        cols = 1 if len(names) == 1 else 2
        for i, name in enumerate(names):
            cell = ctk.CTkFrame(self.grid, fg_color="transparent")
            cell.grid(row=i // cols, column=i % cols, sticky="nsew", padx=3, pady=3)
            lab = ctk.CTkLabel(cell, text=name, text_color="#94a3b8")
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
                                 text_color="#94a3b8")
        self.idle.pack(expand=True)
        self.verdict.configure(text="")
        for w in self.perf.winfo_children():
            w.destroy()

    def tick(self):
        if not self.running:
            return
        live = self.app.cams.running()
        width = 880 if len(self.panes) <= 1 else 470
        for cam in live:
            lab = self.panes.get(cam.name)
            if lab is None:
                continue
            f = cam.overlay_frame(width=width)
            if f is not None:
                img = bgr_to_ctk(f)
                self._imgs[cam.name] = img       # Tk drops unreferenced images
                lab.configure(image=img, text="")

        # Always refresh, even with no live camera: a dead camera must show FAULT,
        # not keep whatever verdict was on screen when it died.
        state, by_cam = self.app.cams.combined()
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in by_cam.items())
        self.verdict.configure(
            text=state if state == infer.PASS else f"{state} — {detail}",
            text_color={infer.PASS: ACC, infer.REJECT: BAD}.get(state, WARN))
        self.app.after(50, self.tick)
        self.update_perf()

    def update_perf(self):
        hz = max(1, int(self.app.settings.get("monitor_hz", 2)))
        now = time.time()
        if now - getattr(self, "_perf_t", 0.0) < 1.0 / hz:
            return
        self._perf_t = now
        rows = self.app.cams.stats()
        for w in self.perf.winfo_children():
            w.destroy()
        hdr = ("CAMERA", "FPS", "READ ms", "INFER ms", "LATENCY ms", "UNSCORED", "SCORED")
        head = ctk.CTkFrame(self.perf, fg_color="transparent")
        head.pack(fill="x")
        for t in hdr:
            ctk.CTkLabel(head, text=t, width=110, anchor="w", text_color=DIM,
                         font=("Segoe UI", 10, "bold")).pack(side="left")
        for s in rows:
            row = ctk.CTkFrame(self.perf, fg_color="transparent")
            row.pack(fill="x")
            # A high unscored share means bottles can pass the camera without
            # the model ever looking at one of their frames.
            drop_col = BAD if s["drop_pct"] > 50 else WARN if s["drop_pct"] > 20 else DIM
            for txt, col in ((s["name"], INK), (s["fps"], INK), (s["read_ms"], DIM),
                             (s["infer_ms"], INK), (s["latency_ms"], DIM),
                             (f"{s['dropped']} ({s['drop_pct']}%)", drop_col),
                             (s["scored"], DIM)):
                ctk.CTkLabel(row, text=str(txt), width=110, anchor="w",
                             text_color=col, font=MONO).pack(side="left")
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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(roi, wraplength=760, justify="left", text_color=DIM,
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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(dup, wraplength=760, justify="left", text_color=DIM,
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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
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
                         font=("Segoe UI", 12, "bold")).pack(side="right")

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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 2))
        self.cmp_note = ctk.CTkLabel(
            body, text_color=DIM, justify="left", wraplength=980,
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
                         font=("Segoe UI", 10, "bold")).pack(side="left")
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
            row = ctk.CTkFrame(self.cmp, fg_color=BG if s != active else "#e3ecfb")
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

        ctk.CTkLabel(parent, text_color=DIM, justify="left", wraplength=1050,
                     text="A webcam asked for 1920x1080 at 30 fps will quietly deliver 7 in poor "
                          "light, and OpenCV reports the number it was asked for. Every column "
                          "here is measured from real frames. Sharpness is the variance of the "
                          "Laplacian: it is what decides whether a meniscus line or label print "
                          "is resolvable at all, which no frame rate can compensate for."
                     ).pack(anchor="w", pady=(0, 8))

        self.out = ctk.CTkFrame(parent, fg_color="transparent")
        self.out.pack(fill="x")
        self.chart = ctk.CTkCanvas(parent, height=210, bg=BG, highlightthickness=0)
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
        self.cam.configure(values=names)
        self.cam.set(names[0])

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
                         font=("Segoe UI", 10, "bold")).pack(side="left")
        best = bench.best_combo(self.rows)
        for r in self.rows:
            row = ctk.CTkFrame(self.out,
                               fg_color="#e3ecfb" if best and r is best else BG)
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
        self.font_val = ctk.CTkLabel(rowf, text="", width=60, font=("Segoe UI", 12, "bold"))
        self.font_val.pack(side="right")
        self.font = ctk.CTkSlider(rowf, from_=0.8, to=1.6, number_of_steps=16,
                                  command=self.set_font)
        self.font.set(float(app.settings.get("font_scale", 1.0)))
        self.font.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(look, text_color=DIM, justify="left", wraplength=760,
                     text="Scales every control with the text, so buttons grow with their labels "
                          "instead of clipping them. Applies immediately and is remembered."
                     ).pack(anchor="w", padx=14, pady=(0, 4))
        quick = ctk.CTkFrame(look, fg_color="transparent")
        quick.pack(anchor="w", padx=14, pady=(0, 12))
        for lab, v in (("Small", 0.9), ("Normal", 1.0), ("Large", 1.2), ("Very large", 1.45)):
            ctk.CTkButton(quick, text=lab, width=90, fg_color="transparent", border_width=1, text_color=ACC,
                          command=lambda v=v: (self.font.set(v), self.set_font(v))
                          ).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(look, text_color=DIM, justify="left", wraplength=760,
                     text="Theme: white and blue. Red, amber and green are kept only where they "
                          "carry meaning - a reject an operator reads across a room, a warning, a "
                          "passing recall - so the theme cannot make a reject look like a pass."
                     ).pack(anchor="w", padx=14, pady=(0, 12))

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
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        return f

    def _number(self, parent, label, value, note):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkLabel(row, text=label, anchor="w", width=330).pack(side="left")
        e = ctk.CTkEntry(row, width=90)
        e.insert(0, str(value))
        e.pack(side="left")
        ctk.CTkLabel(parent, text=note, text_color=DIM, font=("Segoe UI", 10),
                     wraplength=760, justify="left").pack(anchor="w", padx=14, pady=(0, 10))
        return e

    def refresh(self):
        self.font_val.configure(text=f"{float(self.font.get()):.2f}x")
        self.paths.configure(
            text=f"project    {D.PROJECT_DIR}\n"
                 f"images     {D.IMAGE_ROOT}\n"
                 f"labels     {D.LABELS_CSV}\n"
                 f"models     {D.MODELS}\n"
                 f"settings   {D.SETTINGS_JSON}")

    def set_font(self, v):
        self.font_val.configure(text=f"{float(v):.2f}x")
        self.app.apply_font_scale(float(v))

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


def selftest():
    """Build every tab, pump the event loop, exercise the grid, then quit.

    A GUI that imports cleanly can still die on the first widget call. This
    constructs the real window against the real dataset, so a bad geometry call
    or a missing attribute fails here instead of in front of the user.
    """
    app = App()
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

    # Font scaling is one knob and must survive being pushed to both ends.
    before = float(app.settings.get("font_scale", 1.0))
    for v in (0.8, 1.6, before):
        app.tab_settings.set_font(v)
        app.update()
    assert abs(float(app.settings["font_scale"]) - before) < 1e-6

    # Multi-camera panes: build for two sources without opening any device.
    app.tab_live.build_panes(["Camera 0", "Camera 1"])
    app.update()
    assert len(app.tab_live.panes) == 2, app.tab_live.panes
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

    app.cams.stop()
    app.destroy()
    print(f"ok  project {D.PROJECT!r}: {len(app.labels)} images, "
          f"{len(app.defects)} defects, all {len(App.TABS)} tabs built")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        App().mainloop()
