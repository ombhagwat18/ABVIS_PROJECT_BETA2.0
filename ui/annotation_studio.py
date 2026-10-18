"""Annotation Studio -- detection / segmentation labelling with human-reviewed auto-annotation.

Layout (one page of the NexaInspect window): a toolbar with the session status and metric cards on top, then
three zones -- LEFT the dataset context, filters, smart review queues and a paged image list; CENTRE the image
canvas (zoom, pan, fit, overlay toggles); RIGHT the inspector for the selected proposal / box with the review
actions (Accept, Reject, Edit / Correct, Skip, Undo), image actions, bulk actions and export.

Rules this module keeps:
  * It never writes annotations.json itself: every load / validate / save / export goes through vision/annotate.py
    (atomic save) and every review decision through vision/autoannotate.py. labels.csv, train.py and infer.py are
    never touched.
  * A model proposal is not an annotation. It is drawn dashed amber and becomes a box only when a person accepts
    or corrects it; rejected proposals are kept (hidden by default) for the record and undo. An image that still
    holds undecided proposals is not exported.
  * Review actions save at once (atomic) and append to annotation_log.jsonl next to annotations.json; the last
    actions can be undone in this session. Drawing / moving a box by hand also saves when the mouse is released.
  * Proposal generation runs on ONE background thread (Tk widgets are touched only on the main thread, results
    come back through app.post). Results are merged and saved in batches, so a cancel or a crash keeps the work
    already done. Only one generation job at a time, never while the inspection line runs.
  * The page follows the active project. A project that trains a classifier (labels.csv) but also holds an
    annotations.json (the detector dataset, e.g. om_bottle's bootstrapped defect boxes) opens it for review.

`AnnotationTab` keeps the `(app, parent)` + `refresh()` contract of every page.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from vision import annotate as A
from vision import autoannotate as AA
from vision import dataset as D
from ui import components as C
from ui import theme
from vision import verdict as VD

# The shared palette (theme.py). Imported from theme, not app, to avoid an import cycle.
from ui.theme import (ACC, ACC_H, ACC_SOFT, ACC_T, BAD, BG, DIM, GOOD, INFO, INK, LINE, MUTED, PANEL,  # noqa: F401
                      PANEL_2, WARN)

CANVAS_W, CANVAS_H = 520, 400      # initial size; the canvas then follows the window
HANDLE = 6                         # canvas px, half-width of a resize handle
MIN_BOX = 0.01                     # normalized -- boxes/drags smaller than this are discarded, not saved
PAGE = 40                          # image-list rows per page (thumbnails are loaded for one page only)
THUMB = 44
BATCH = 8                          # generation results merged + saved per batch
UNDO_MAX = 50

# Overlay colours: pending proposal amber (dashed), accepted green, corrected blue-info, rejected red (dashed,
# hidden by default), selected = the accent blue with handles. Every state also has a word in the inspector.
COL = {"prop": WARN, "box": GOOD, "corrected": INFO, "rej": BAD, "sel": ACC, "manual": GOOD, "poly": GOOD}
STATE_FILTERS = (("All images", "all"), ("Pending proposals", "pending"), ("Accepted", "accepted"),
                 ("Corrected", "corrected"), ("Rejected", "rejected"), ("Failed", "failed"),
                 ("Skipped for later", "skipped"), ("Unreviewed", "unreviewed"), ("Reviewed", "reviewed"),
                 ("Not proposed yet", "not_proposed"))
QUEUES = ("Dataset order", "Least confident", "Missing-part candidates", "Doubtful part / hard negatives")
SCOPES = ("Images not proposed yet", "Failed images (retry)", "Current list (replaces pending proposals)")
STATE_COL = {"failed": BAD, "pending": WARN, "skipped": MUTED, "corrected": INFO, "accepted": GOOD,
             "rejected": BAD, "reviewed": GOOD, "annotated": GOOD}


class AnnotationTab:
    """The 'Annotate' page. Meaningful for a project with an annotations.json (detection / segmentation)."""

    def __init__(self, app, parent):
        self.app = app
        self.parent = parent

        self.task: str | None = None
        self.ann_path: Path | None = None
        self.data: dict | None = None
        self.images: list[str] = []          # every image of the dataset (dataset order)
        self.view: list[str] = []            # the filtered / queued list shown on the left
        self.page = 0
        self.idx = 0
        self.rel: str | None = None
        self.dirty = False                   # unsaved edits made outside the UI actions (those save at once)
        self.image_root: Path | None = None  # None -> falls back to D.IMAGE_ROOT
        self.meta: dict[str, dict] = {}      # rel -> {"scene_id":.., "split":..}, optional
        self.external = False                # load_external(): an explicit image set, not the project's images/
        self.autosave = True

        self.img_bgr = None
        self.img_rgb = None
        self.img_w = self.img_h = 0
        self.cw, self.ch = CANVAS_W, CANVAS_H
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.scale = 1.0
        self.off_x = self.off_y = 0.0
        self.photo = None                    # keep alive -- Tk drops unreferenced PhotoImages

        self.selected: tuple | None = None   # ("box", i) | ("prop", i) | ("rej", i) | ("poly", i)
        self.drag: dict | None = None
        self.poly_points: list[float] = []   # in-progress polygon, flat [x,y,...]
        self._pan_from = None

        self.undo_stack: list = []           # [(label, {rel: entry-or-None before the action})]
        self.job: dict | None = None         # the one proposal-generation job
        self._q: queue.Queue = queue.Queue() # results from threads when there is no app (self-test)
        self._thumbs: dict = {}              # rel -> PhotoImage (current page only)
        self._thumb_token = 0
        self._rows: dict = {}
        self._row_imgs: list = []
        self._error = ""

        self._build(parent)
        top = parent.winfo_toplevel()
        top.bind_all("<KeyPress>", self._on_key, add="+")

    # ================================================================== building
    def _build(self, parent):
        self.notice = ctk.CTkLabel(parent, text="", text_color=DIM, font=(theme.UI, 15), wraplength=700,
                                   justify="left")
        self.init_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.body = ctk.CTkFrame(parent, fg_color="transparent")   # packed only when there is something to annotate

        self._build_toolbar(self.body)
        cols = ctk.CTkFrame(self.body, fg_color="transparent")
        cols.pack(fill="both", expand=True)
        self._build_left(cols)
        self._build_right(cols)
        self._build_centre(cols)

    def _build_toolbar(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=6, border_width=1, border_color=LINE)
        bar.pack(fill="x", pady=(0, 6))
        # row 1: what is being annotated and with which model
        row = ctk.CTkFrame(bar, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 4))
        self.pill = C.StatusPill(row)
        self.pill.pack(side="left", padx=(0, 10))
        self.ctx_lbl = ctk.CTkLabel(row, text="-", font=(theme.UI, 13, "bold"), text_color=INK, anchor="w")
        self.ctx_lbl.pack(side="left", padx=(0, 12))
        ctk.CTkLabel(row, text="Model", text_color=DIM).pack(side="left")
        self.model_menu = ctk.CTkOptionMenu(row, values=["-"], width=200, dynamic_resizing=False)
        self.model_menu.pack(side="left", padx=(4, 8))
        ctk.CTkLabel(row, text="Min conf", text_color=DIM).pack(side="left")
        self.conf_entry = ctk.CTkEntry(row, width=48)
        self.conf_entry.insert(0, "0.25")
        self.conf_entry.pack(side="left", padx=(4, 8))
        self.scope_menu = ctk.CTkOptionMenu(row, values=list(SCOPES), width=210, dynamic_resizing=False)
        self.scope_menu.pack(side="left")
        # row 2: run controls, progress, export
        run = ctk.CTkFrame(bar, fg_color="transparent")
        run.pack(fill="x", padx=10, pady=(0, 4))
        self.auto_btn = C.button(run, "Generate proposals", self.auto_annotate, "primary", 140,
                                 tip="Run the selected detector on the chosen images. Results are PROPOSALS:\n"
                                     "nothing becomes a label until you accept it.")
        self.auto_btn.pack(side="left", padx=(0, 4))
        self.pause_btn = C.button(run, "Pause", self.toggle_pause, width=62)
        self.pause_btn.pack(side="left", padx=(0, 4))
        self.cancel_btn = C.button(run, "Cancel", self.cancel_job, "danger", 62)
        self.cancel_btn.pack(side="left", padx=(0, 10))
        self.export_seg_btn = C.button(run, "Export polygons", self.export_segmentation, width=112)
        self.export_seg_btn.pack(side="right")
        self.export_det_btn = C.button(run, "Export approved", self.export_detection, "success", 120,
                                       tip="YOLO detection labels of ACCEPTED / corrected boxes only.\n"
                                           "Images with undecided proposals are left out.")
        self.export_det_btn.pack(side="right", padx=(0, 4))
        self.prog_lbl = ctk.CTkLabel(run, text="", text_color=DIM, font=theme.TINY, anchor="e", width=250)
        self.prog_lbl.pack(side="right", padx=(0, 10))
        self.prog_bar = ctk.CTkProgressBar(run, height=8, width=60)
        self.prog_bar.set(0)
        self.prog_bar.pack(side="left", fill="x", expand=True)
        cards = ctk.CTkFrame(bar, fg_color="transparent")
        cards.pack(fill="x", padx=10, pady=(0, 8))
        self.cards = {}
        for key, cap in (("images", "Images"), ("proposed_images", "Proposed"), ("pending", "Pending"),
                         ("accepted", "Accepted"), ("corrected", "Corrected"), ("rejected", "Rejected"),
                         ("failed", "Failed"), ("reviewed", "Reviewed"), ("decided_pct", "Decided")):
            card = C.MetricCard(cards, cap)
            card.pack(side="left", padx=(0, 6))
            self.cards[key] = card

    def _build_left(self, parent):
        left = ctk.CTkFrame(parent, fg_color=PANEL, width=290, corner_radius=6, border_width=1, border_color=LINE)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)
        C.section(left, "Dataset")
        self.ds_lbl = ctk.CTkLabel(left, text="", text_color=DIM, font=theme.TINY, justify="left", anchor="w",
                                   wraplength=262)
        self.ds_lbl.pack(fill="x", padx=12)
        C.section(left, "Filter")
        f = ctk.CTkFrame(left, fg_color="transparent")
        f.pack(fill="x", padx=12)
        self.state_menu = ctk.CTkOptionMenu(f, values=[s for s, _ in STATE_FILTERS], width=262,
                                            command=lambda _v: self.apply_filter(), dynamic_resizing=False)
        self.state_menu.pack(fill="x", pady=2)
        self.queue_menu = ctk.CTkOptionMenu(f, values=list(QUEUES), width=262, command=lambda _v: self.apply_filter(),
                                            dynamic_resizing=False)
        self.queue_menu.pack(fill="x", pady=2)
        g = ctk.CTkFrame(f, fg_color="transparent")
        g.pack(fill="x", pady=2)
        self.cls_filter = ctk.CTkOptionMenu(g, values=["all classes"], width=130,
                                            command=lambda _v: self.apply_filter(), dynamic_resizing=False)
        self.cls_filter.pack(side="left")
        ctk.CTkLabel(g, text="conf", text_color=DIM).pack(side="left", padx=(6, 2))
        self.lo_entry = ctk.CTkEntry(g, width=46)
        self.lo_entry.insert(0, "0")
        self.lo_entry.pack(side="left")
        ctk.CTkLabel(g, text="-", text_color=DIM).pack(side="left", padx=2)
        self.hi_entry = ctk.CTkEntry(g, width=46)
        self.hi_entry.insert(0, "1")
        self.hi_entry.pack(side="left")
        for e in (self.lo_entry, self.hi_entry):
            e.bind("<Return>", lambda _e: self.apply_filter())
        self.queue_note = ctk.CTkLabel(left, text="", text_color=DIM, font=theme.TINY, justify="left", anchor="w",
                                       wraplength=262)
        self.queue_note.pack(fill="x", padx=12, pady=(2, 0))
        hdr = ctk.CTkFrame(left, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(hdr, text="IMAGES", text_color=DIM, font=theme.CAPS).pack(side="left")
        self.page_lbl = ctk.CTkLabel(hdr, text="", text_color=DIM, font=theme.TINY)
        self.page_lbl.pack(side="right")
        # The controls under the list are packed to the BOTTOM first, so a short window shrinks the list, never them.
        bottom = ctk.CTkFrame(left, fg_color="transparent")
        bottom.pack(side="bottom", fill="x")
        self.imglist = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.imglist.pack(fill="both", expand=True, padx=4)
        pg = ctk.CTkFrame(bottom, fg_color="transparent")
        pg.pack(fill="x", padx=10, pady=(4, 0))
        C.button(pg, "< page", lambda: self.set_page(self.page - 1), width=70).pack(side="left")
        C.button(pg, "page >", lambda: self.set_page(self.page + 1), width=70).pack(side="right")
        nav = ctk.CTkFrame(bottom, fg_color="transparent")
        nav.pack(fill="x", padx=10, pady=4)
        C.button(nav, "< Prev", self.prev_image, width=80, tip="Previous image  (P)").pack(side="left")
        C.button(nav, "Next >", self.next_image, width=80, tip="Next image  (N)").pack(side="right")
        nav2 = ctk.CTkFrame(bottom, fg_color="transparent")
        nav2.pack(fill="x", padx=10)
        C.button(nav2, "Next pending", self.goto_next_pending, width=124).pack(side="left")
        C.button(nav2, "Next unannotated", self.goto_next_unannotated, width=124).pack(side="right")
        self.progress_lbl = ctk.CTkLabel(bottom, text="", text_color=DIM, font=theme.TINY, justify="left",
                                         anchor="w", wraplength=262)
        self.progress_lbl.pack(fill="x", padx=12, pady=(4, 8))

    def _build_centre(self, parent):
        center = ctk.CTkFrame(parent, fg_color="transparent")
        center.pack(side="left", fill="both", expand=True)
        top = ctk.CTkFrame(center, fg_color=PANEL, corner_radius=6, border_width=1, border_color=LINE)
        top.pack(fill="x", pady=(0, 4))
        self.fname_lbl = ctk.CTkLabel(top, text="-", font=(theme.UI, 13, "bold"))
        self.fname_lbl.pack(side="left", padx=10, pady=6)
        self.meta_lbl = ctk.CTkLabel(top, text="", text_color=DIM)
        self.meta_lbl.pack(side="left", padx=6)
        self.reviewed_lbl = ctk.CTkLabel(top, text="", text_color=DIM)
        self.reviewed_lbl.pack(side="left", padx=6)
        self.zoom_lbl = ctk.CTkLabel(top, text="", text_color=DIM, font=theme.TINY)
        self.zoom_lbl.pack(side="right", padx=10)
        view = ctk.CTkFrame(center, fg_color="transparent")
        view.pack(fill="x", pady=(0, 4))
        C.button(view, "Fit", self.fit, width=50, tip="Fit image to canvas  (F)").pack(side="left")
        C.button(view, "-", lambda: self.zoom_by(1 / 1.25), width=34, tip="Zoom out  (-)").pack(side="left", padx=2)
        C.button(view, "+", lambda: self.zoom_by(1.25), width=34, tip="Zoom in  (+), or the mouse wheel").pack(
            side="left", padx=2)
        C.button(view, "100%", self.actual_size, width=56, tip="1 image pixel = 1 screen pixel").pack(side="left",
                                                                                                         padx=(2, 10))
        self.show = {}
        for key, text, on in (("prop", "Proposals", True), ("box", "Accepted", True), ("rej", "Rejected", False),
                              ("labels", "Labels", True), ("conf", "Confidence", True)):
            v = ctk.BooleanVar(value=on)
            ctk.CTkCheckBox(view, text=text, variable=v, width=20, checkbox_width=16, checkbox_height=16,
                            command=self._redraw).pack(side="left", padx=4)
            self.show[key] = v
        self.canvas = tk.Canvas(center, width=CANVAS_W, height=CANVAS_H, bg=theme.VIDEO_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self._right_press)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<MouseWheel>", self._wheel)
        actionbar = ctk.CTkFrame(center, fg_color="transparent")
        actionbar.pack(fill="x", pady=(4, 0))
        C.button(actionbar, "Delete selected", self.delete_selected, "danger", 120, tip="Delete  (Del)").pack(
            side="left", padx=(0, 4))
        self.finish_btn = C.button(actionbar, "Finish polygon", self.finish_polygon, width=120)
        self.finish_btn.pack(side="left", padx=(0, 4))
        self.cancel_poly_btn = C.button(actionbar, "Cancel polygon", self.cancel_polygon, width=120)
        self.cancel_poly_btn.pack(side="left")
        self.hint = ctk.CTkLabel(actionbar, text="", text_color=DIM, font=theme.TINY)
        self.hint.pack(side="right")

    def _build_right(self, parent):
        right = ctk.CTkFrame(parent, fg_color=PANEL, width=300, corner_radius=6, border_width=1, border_color=LINE)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        inner = ctk.CTkScrollableFrame(right, fg_color="transparent")
        inner.pack(fill="both", expand=True)
        C.section(inner, "Inspector")
        self.insp_state = ctk.CTkLabel(inner, text="nothing selected", font=(theme.UI, 14, "bold"), anchor="w")
        self.insp_state.pack(fill="x", padx=12)
        self.insp_lbl = ctk.CTkLabel(inner, text="", text_color=INK, font=theme.TINY, justify="left", anchor="w",
                                     wraplength=262)
        self.insp_lbl.pack(fill="x", padx=12, pady=(2, 0))
        self.warn_lbl = ctk.CTkLabel(inner, text="", text_color=BAD, font=theme.TINY, justify="left", anchor="w",
                                     wraplength=262)
        self.warn_lbl.pack(fill="x", padx=12)
        g = ctk.CTkFrame(inner, fg_color="transparent")
        g.pack(fill="x", padx=12, pady=(6, 0))
        self.accept_sel_btn = C.button(g, "Accept", self.accept_selected, "success", 124, tip="Accept proposal  (A)")
        self.accept_sel_btn.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="ew")
        self.reject_sel_btn = C.button(g, "Reject", self.reject_selected, "danger", 124, tip="Reject proposal  (R)")
        self.reject_sel_btn.grid(row=0, column=1, pady=2, sticky="ew")
        self.edit_btn = C.button(g, "Edit / correct", self.edit_selected, width=124,
                                 tip="Turn the proposal into a box you can move, resize or re-class  (E).\n"
                                     "The model's original is kept; a change is recorded as CORRECTED.")
        self.edit_btn.grid(row=1, column=0, padx=(0, 4), pady=2, sticky="ew")
        self.skip_btn = C.button(g, "Skip for later", self.skip_image, width=124, tip="Skip image  (S)")
        self.skip_btn.grid(row=1, column=1, pady=2, sticky="ew")
        self.undo_btn = C.button(g, "Undo", self.undo, width=124, tip="Undo the last action  (Ctrl+Z)")
        self.undo_btn.grid(row=2, column=0, padx=(0, 4), pady=2, sticky="ew")
        C.button(g, "Next pending >", self.goto_next_pending, "ghost", 124).grid(row=2, column=1, pady=2,
                                                                                  sticky="ew")
        C.section(inner, "Class")
        cr = ctk.CTkFrame(inner, fg_color="transparent")
        cr.pack(fill="x", padx=12)
        self.class_menu = ctk.CTkOptionMenu(cr, values=["-"], width=170, dynamic_resizing=False)
        self.class_menu.pack(side="left")
        C.button(cr, "Apply", self.apply_class, width=70,
                 tip="Change the selected box's / proposal's class (a proposal becomes a CORRECTED box).\n"
                     "The same list is the class for new boxes you draw.").pack(side="right")
        C.section(inner, "This image")
        self.pred_lbl = ctk.CTkLabel(inner, text="", font=(theme.UI, 13, "bold"), wraplength=262, justify="left",
                                     anchor="w")
        self.pred_lbl.pack(fill="x", padx=12)
        h = ctk.CTkFrame(inner, fg_color="transparent")
        h.pack(fill="x", padx=12, pady=(4, 0))
        self.accept_btn = C.button(h, "Accept all", self.accept_proposals, "success", 124)
        self.accept_btn.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="ew")
        C.button(h, "Reject all", self.reject_proposals, "danger", 124).grid(row=0, column=1, pady=2, sticky="ew")
        C.button(h, "Mark reviewed", self.mark_reviewed, "primary", 252,
                 tip="You vouch for this image's boxes (no boxes = a true background image).\n"
                     "Undecided proposals must be decided first.").grid(row=1, column=0, columnspan=2, pady=2,
                                                                       sticky="ew")
        C.section(inner, "Bulk")
        b = ctk.CTkFrame(inner, fg_color="transparent")
        b.pack(fill="x", padx=12)
        C.button(b, "Bulk accept...", lambda: self.bulk_dialog("accept"), width=124).pack(side="left")
        C.button(b, "Bulk reject...", lambda: self.bulk_dialog("reject"), width=124).pack(side="right")
        C.section(inner, "File")
        fl = ctk.CTkFrame(inner, fg_color="transparent")
        fl.pack(fill="x", padx=12)
        C.button(fl, "Save", self.save, "primary", 80).pack(side="left")
        C.button(fl, "Reload", self.reload_annotations, width=80).pack(side="left", padx=4)
        C.button(fl, "History", self.show_history, width=80).pack(side="left")
        self.status = ctk.CTkLabel(inner, text="", text_color=DIM, font=theme.TINY, wraplength=262, justify="left",
                                   anchor="w")
        self.status.pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(inner, text="Keys: A accept  R reject  E edit  S skip  N / P next / prev  F fit  + / - zoom  "
                                 "Del delete  Esc cancel  Ctrl+Z undo.  Wheel = zoom, right / middle drag = pan.",
                     text_color=MUTED, font=theme.TINY, wraplength=262, justify="left", anchor="w").pack(
            fill="x", padx=12, pady=(4, 10))

    # ================================================================== threading helpers
    def _post(self, fn):
        """Run fn on the Tk thread (from any thread)."""
        if self.app is not None and hasattr(self.app, "post"):
            self.app.post(fn)
        else:
            self._q.put(fn)

    def _drain(self) -> int:
        """Run what threads queued (only used when there is no App, e.g. the self-test)."""
        n = 0
        while True:
            try:
                fn = self._q.get_nowait()
            except queue.Empty:
                return n
            fn()
            n += 1

    # ================================================================== refresh / loading a dataset
    def refresh(self):
        """Called by App.reload() on every project switch / data change, as every other page."""
        if self.external:
            return
        self.image_root = D.IMAGE_ROOT
        self.meta = {}
        task = D.project_task(D.PROJECT) if D.PROJECT else D.DEFAULT_TASK
        self.task = task
        self.body.pack_forget()
        self.init_frame.pack_forget()
        self.ann_path = D.PROJECT_DIR / "annotations.json"
        if task not in ("detection", "segmentation") and not self.ann_path.exists():
            self.notice.configure(
                text=("Annotation mode is not required for this project's task "
                      f"({task!r}).\n\nThis project trains a classifier from labels.csv, "
                      "labelled in the Label tab. The Annotate tab applies only to "
                      "projects whose task is \"detection\" or \"segmentation\", or that hold a detector "
                      "dataset (annotations.json)."))
            self.notice.pack(pady=60)
            return
        self.notice.pack_forget()
        if not self.ann_path.exists():
            self._show_init_prompt(task)
            return
        try:
            self.data = A.load_annotations(self.ann_path)
        except (A.AnnotationError, ValueError, OSError) as e:
            self.notice.configure(text=f"annotations.json could not be loaded:\n{e}", text_color=BAD)
            self.notice.pack(pady=60)
            return
        self.task = self.data["task"]
        self.dirty = False
        self.undo_stack.clear()
        imgs = D.scan_images()
        known = set(imgs)
        self.images = imgs + [r for r in self.data["images"] if r not in known]
        self._open(task_note=(f"Project task {task!r}: this annotations.json is its detector dataset. labels.csv is "
                              "never changed here." if task not in ("detection", "segmentation") else ""))

    def _show_init_prompt(self, task):
        for w in self.init_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.init_frame,
                     text=f"This project's task is {task!r} but no annotations.json exists yet.",
                     font=(theme.UI, 16, "bold")).pack(pady=(60, 6))
        ctk.CTkLabel(self.init_frame,
                     text="Enter class names, comma-separated, to initialize it. "
                          "This creates annotations.json only for THIS project.",
                     text_color=DIM, wraplength=500, justify="center").pack(pady=(0, 10))
        entry = ctk.CTkEntry(self.init_frame, width=380, placeholder_text="e.g. scratch, dent, missing_cap")
        entry.pack(pady=6)

        def do_init():
            names = [c.strip() for c in entry.get().split(",") if c.strip()]
            if not names:
                messagebox.showinfo("Class names required", "Enter at least one class name.")
                return
            try:
                data = A.init_annotations(task, names)
                A.save_annotations(self.ann_path, data)
            except A.AnnotationError as e:
                messagebox.showerror("Cannot initialize", str(e))
                return
            self.refresh()

        C.button(self.init_frame, "Initialize annotations.json", do_init, "primary").pack(pady=8)
        self.init_frame.pack(fill="both", expand=True)

    def load_external(self, image_root: Path, ann_path: Path, task: str, classes: list[str],
                      images: list[str], meta: dict[str, dict] | None = None):
        """Point this page at an explicit image set instead of the active project's images/ folder -- used for
        Stage 2 detection annotation, where the images live under stage2_dataset/. Never touches dataset.py's
        project globals, review_manifest.csv, split.json, split_manifest.csv, or Stage 1.

        `ann_path` is created fresh (via annotate.init_annotations) only if it doesn't already exist -- an existing
        file's annotations are always loaded and preserved, never overwritten by this call."""
        self.notice.pack_forget()
        self.init_frame.pack_forget()
        self.external = True
        self.image_root = Path(image_root)
        self.ann_path = Path(ann_path)
        self.task = task
        self.meta = meta or {}
        if not self.ann_path.exists():
            A.save_annotations(self.ann_path, A.init_annotations(task, classes))
        self.data = A.load_annotations(self.ann_path)
        self.dirty = False
        self.undo_stack.clear()
        self.images = list(images)
        self.idx = 0
        self._open()

    def _open(self, task_note: str = ""):
        task = self.task
        self._populate_classes()
        self._populate_models()
        self.export_det_btn.configure(state="normal" if task == "detection" else "disabled")
        self.export_seg_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.finish_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.cancel_poly_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.auto_btn.configure(state="normal" if task == "detection" else "disabled")
        self.hint.configure(
            text=("drag = new box  |  click = select  |  drag box / corners = edit"
                  if task == "detection" else
                  "click to add polygon points, right-click or 'Finish polygon' to close"))
        where = "external set" if self.external else (D.PROJECT or "-")
        self.ctx_lbl.configure(text=f"{where}  |  {task}")
        self.ds_lbl.configure(text=f"{self.ann_path}\n{len(self.images)} images  ·  {len(self.data['classes'])} classes"
                                   + (f"\n{task_note}" if task_note else ""))
        self.body.pack(fill="both", expand=True)
        self.apply_filter(keep=False)
        self._update_metrics()
        if self.images:
            self.idx = min(self.idx, len(self.images) - 1)
            first = self.view[0] if self.view else self.images[self.idx]
            self.load_image(first)
        else:
            self.rel = None
            self.fname_lbl.configure(text="-")
            self.status.configure(text="No images found under this project's images/ folder yet.")
            self.canvas.delete("all")

    def _populate_classes(self):
        values = list(self.data["classes"]) or ["-"]
        self.class_menu.configure(values=values)
        self.class_menu.set(values[0])
        self.cls_filter.configure(values=["all classes"] + list(self.data["classes"]))
        self.cls_filter.set("all classes")

    def _populate_models(self):
        try:
            settings = D.load_settings()
        except Exception:                                        # noqa: BLE001
            settings = {}
        root = Path(__file__).resolve().parent.parent
        self._models = AA.detector_choices(root, settings)
        labels = [m[0] for m in self._models] or ["no detector weights found"]
        self.model_menu.configure(values=labels)
        self.model_menu.set(labels[0])
        self.conf_entry.delete(0, "end")
        self.conf_entry.insert(0, str(settings.get("detector_conf", 0.25)))

    # ================================================================== filters, queues, the image list
    def _filter_args(self):
        state = dict(STATE_FILTERS).get(self.state_menu.get(), "all")
        cls = self.cls_filter.get()
        cls = None if cls == "all classes" else cls
        try:
            lo, hi = float(self.lo_entry.get() or 0), float(self.hi_entry.get() or 1)
        except ValueError:
            lo, hi = 0.0, 1.0
        return state, cls, max(0.0, lo), min(1.0, hi)

    def apply_filter(self, keep: bool = True):
        if self.data is None:
            return
        state, cls, lo, hi = self._filter_args()
        q = self.queue_menu.get()
        if q == "Least confident":
            order = AA.order_for_review(self.data, self.images)
            note = "Unreviewed images, lowest proposal confidence first (an image with no proposal first of all)."
        elif q == "Missing-part candidates":
            order = AA.missing_part_queue(self.data, self.images)
            note = ("Bottle proposed, cap or label not: CANDIDATES for a missing part - or occlusion, angle, "
                    "a missed detection. Review; nothing here is a confirmed defect.")
        elif q == "Doubtful part / hard negatives":
            order = AA.doubtful_part_queue(self.data, self.images)
            note = ("A cap / label proposed at 0.25-0.80: often a hard negative (e.g. a tamper ring read as a cap). "
                    "Reject what is not the part.")
        else:
            order, note = list(self.images), ""
        self.view = [r for r in order if AA.matches(self.data["images"].get(r), state, cls, lo, hi)]
        self.queue_note.configure(text=note)
        if keep and self.rel in self.view:
            self.page = self.view.index(self.rel) // PAGE
        else:
            self.page = 0
        self._rebuild_image_list()

    def set_page(self, page: int):
        n = max(1, (len(self.view) + PAGE - 1) // PAGE)
        self.page = min(max(0, page), n - 1)
        self._rebuild_image_list()

    def _row_text(self, rel):
        e = (self.data or {}).get("images", {}).get(rel) or {}
        word = AA.state_word(e)
        props = e.get("proposals") or []
        bits = []
        if props:
            cs = [p.get("conf", 0) for p in props]
            bits.append(f"{len(props)} pending  {min(cs):.2f}-{max(cs):.2f}")
        nb = len(e.get("boxes") or []) + len(e.get("polygons") or [])
        if nb:
            bits.append(f"{nb} boxes")
        if e.get("failed"):
            bits.append(str(e["failed"].get("reason", ""))[:40])
        return word, "  ·  ".join(bits)

    def _rebuild_image_list(self):
        for w in self.imglist.winfo_children():
            w.destroy()
        self._rows.clear()
        self._row_imgs.clear()
        self._thumbs.clear()
        start = self.page * PAGE
        rows = self.view[start:start + PAGE]
        n_pages = max(1, (len(self.view) + PAGE - 1) // PAGE)
        self.page_lbl.configure(text=f"{len(self.view)} shown  ·  page {self.page + 1}/{n_pages}")
        for rel in rows:
            row = tk.Frame(self.imglist, bg=ACC_SOFT if rel == self.rel else PANEL, cursor="hand2")
            row.pack(fill="x", pady=1, padx=2)
            th = tk.Label(row, bg=theme.VIDEO_BG, width=THUMB // 8, height=THUMB // 16)
            th.pack(side="left", padx=(4, 6), pady=3)
            txt = tk.Frame(row, bg=row["bg"])
            txt.pack(side="left", fill="x", expand=True)
            name = tk.Label(txt, text=Path(rel).name, bg=row["bg"], fg=INK, font=(theme.UI, 10, "bold"), anchor="w")
            name.pack(fill="x")
            word, info = self._row_text(rel)
            st = tk.Label(txt, text=f"● {word.upper()}", bg=row["bg"], fg=STATE_COL.get(word, DIM),
                          font=(theme.UI, 9, "bold"), anchor="w")
            st.pack(fill="x")
            inf = tk.Label(txt, text=info, bg=row["bg"], fg=DIM, font=(theme.UI, 9), anchor="w")
            inf.pack(fill="x")
            for w in (row, th, txt, name, st, inf):
                w.bind("<Button-1>", lambda e, p=rel: self.goto_image(p))
            self._rows[rel] = (row, th, txt, name, st, inf)
        self._update_progress_label()
        self._load_thumbs(rows)

    def _update_row(self, rel):
        r = self._rows.get(rel)
        if not r:
            return
        word, info = self._row_text(rel)
        r[4].configure(text=f"● {word.upper()}", fg=STATE_COL.get(word, DIM))
        r[5].configure(text=info)

    def _load_thumbs(self, rows):
        """One page of thumbnails, decoded on a worker thread; PhotoImages made on the Tk thread."""
        if self.app is None or not rows:
            return
        self._thumb_token += 1
        token, root = self._thumb_token, self.image_root or D.IMAGE_ROOT
        use_cache = (not self.external) and root == D.IMAGE_ROOT

        def work():
            import numpy as np
            for rel in rows:
                if token != self._thumb_token:
                    return
                img = None
                try:
                    if use_cache:
                        b = D.thumbnail(rel)
                        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
                    else:
                        img = D.imread(Path(root) / rel)
                except Exception:                                # noqa: BLE001 - a thumbnail is never essential
                    img = None
                if img is None:
                    continue
                h, w = img.shape[:2]
                s = THUMB / max(h, w)
                small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                self._post(lambda r=rel, p=pil: self._set_thumb(token, r, p))

        threading.Thread(target=work, daemon=True, name="annotate-thumbs").start()

    def _set_thumb(self, token, rel, pil):
        if token != self._thumb_token or rel not in self._rows:
            return
        ph = ImageTk.PhotoImage(pil)
        self._thumbs[rel] = ph
        self._rows[rel][1].configure(image=ph, width=THUMB, height=THUMB)

    def _update_progress_label(self):
        if self.data is None:
            self.progress_lbl.configure(text="")
            return
        c = A.progress_counts(self.data, self.images)
        self.progress_lbl.configure(text=f"total {c['total']}  ·  pending {c['pending']}  ·  "
                                         f"annotated {c['annotated']}  ·  reviewed {c['reviewed']}")

    def _update_metrics(self):
        if self.data is None:
            return
        c = AA.counts(self.data, self.images)
        for k, card in self.cards.items():
            v = c.get(k, 0)
            colour = {"pending": WARN, "accepted": GOOD, "corrected": INFO, "rejected": BAD,
                      "failed": BAD if v else INK}.get(k, INK)
            card.set(f"{v:.0f}%" if k == "decided_pct" else v, colour if v else INK)
        self._update_progress_label()
        if self.job is None and self.pill.state not in ("ERROR", "EXPORTING"):
            self.pill.set("REVIEWING" if c["pending"] else "IDLE")

    def _highlight_row(self, rel):
        for p, r in self._rows.items():
            bg = ACC_SOFT if p == rel else PANEL
            for w in (r[0], r[2], r[3], r[4], r[5]):
                w.configure(bg=bg)

    # ================================================================== navigation
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno("Unsaved changes", "Discard unsaved annotation changes for this image?")

    def goto_image(self, rel):
        if rel == self.rel:
            return
        if not self._confirm_discard():
            return
        self.dirty = False
        if rel in self.images:
            self.idx = self.images.index(rel)
        self.load_image(rel)

    def _step(self, d):
        if not self.images or not self._confirm_discard():
            return
        self.dirty = False
        seq = self.view if (self.view and self.rel in self.view) else self.images
        i = seq.index(self.rel) if self.rel in seq else -1
        j = min(max(i + d, 0), len(seq) - 1)
        self.idx = self.images.index(seq[j]) if seq[j] in self.images else self.idx
        self.load_image(seq[j])

    def next_image(self):
        self._step(1)

    def prev_image(self):
        self._step(-1)

    def goto_next_pending(self):
        if self.data is None:
            return
        seq = self.view or self.images
        start = seq.index(self.rel) + 1 if self.rel in seq else 0
        for k in range(len(seq)):
            r = seq[(start + k) % len(seq)]
            if r != self.rel and (self.data["images"].get(r) or {}).get("proposals"):
                return self.goto_image(r)
        messagebox.showinfo("Next pending", "No other image in this list has undecided proposals.")

    def goto_next_unannotated(self):
        if self.data is None:
            return
        rel = A.next_unannotated(self.images, self.data, after=self.rel)
        if rel is None:
            messagebox.showinfo("Next Unannotated", "No pending (unannotated) images left.")
            return
        self.goto_image(rel)

    def load_image(self, rel):
        root = self.image_root if self.image_root is not None else D.IMAGE_ROOT
        img = D.imread(root / rel)
        if img is None:
            self.status.configure(text=f"Cannot read image: {rel}", text_color=BAD)
            messagebox.showerror("Cannot read image", rel)
            return
        self.rel = rel
        self.img_bgr = img
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = img.shape[:2]
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self._update_view()
        self.selected = None
        self.drag = None
        self.poly_points = []
        self.fname_lbl.configure(text=rel)
        m = self.meta.get(rel, {})
        self.meta_lbl.configure(text=f"scene {m['scene_id']} · {m['split']}" if m else "")
        self._update_image_header()
        if self.rel in self.view and not (self.page * PAGE <= self.view.index(self.rel) < (self.page + 1) * PAGE):
            self.page = self.view.index(self.rel) // PAGE
            self._rebuild_image_list()
        self._highlight_row(rel)
        self._redraw()

    def _update_image_header(self):
        entry = self.data["images"].get(self.rel, {}) if self.data and self.rel else {}
        reviewed = bool(entry.get("reviewed"))
        word = AA.state_word(entry)
        self.reviewed_lbl.configure(text=("reviewed" if reviewed else "unreviewed") + f"  ·  {word}",
                                    text_color=GOOD if reviewed else WARN)

    def _show_prediction(self, entry):
        """'PREDICTED: GOOD / DEFECT: Missing cap / NO BOTTLE' for detection projects, never a label by itself."""
        if self.task != "detection" or self.img_bgr is None:
            self.pred_lbl.configure(text="")
            return
        try:
            kind, defects, src = AA.predict(entry, (self.img_w, self.img_h), D.load_config().get("inspection"))
        except Exception as e:                                   # noqa: BLE001 - a display line must not break drawing
            self.pred_lbl.configure(text=f"prediction unavailable: {type(e).__name__}", text_color=DIM)
            return
        if not kind:
            self.pred_lbl.configure(text="no boxes yet: generate proposals to get a prediction", text_color=DIM)
            return
        words = {"GOOD": "GOOD", "NO BOTTLE": "NO BOTTLE FOUND"}.get(kind) or \
            "DEFECT: " + ", ".join(VD.pretty(d) for d in defects)
        who = "model proposal - not reviewed" if src == "proposals" else "from your boxes"
        self.pred_lbl.configure(text=f"PREDICTED: {words}\n({who})",
                                text_color={"GOOD": GOOD, "DEFECT": BAD}.get(kind, WARN))

    # ================================================================== view: zoom / pan / coordinates
    def _on_resize(self, e):
        if e.width < 20 or e.height < 20:
            return
        self.cw, self.ch = e.width, e.height
        self._update_view()
        self._redraw()

    def _fit_scale(self):
        if not self.img_w:
            return 1.0
        return min(self.cw / self.img_w, self.ch / self.img_h) * 0.98

    def _update_view(self):
        self.scale = self._fit_scale() * self.zoom
        self.off_x = (self.cw - self.img_w * self.scale) / 2 + self.pan_x
        self.off_y = (self.ch - self.img_h * self.scale) / 2 + self.pan_y
        self.zoom_lbl.configure(text=f"{self.img_w} x {self.img_h} px   ·   zoom {self.scale * 100:.0f}%"
                                if self.img_w else "")

    def fit(self):
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self._update_view()
        self._redraw()

    def actual_size(self):
        f = self._fit_scale()
        self.zoom = 1.0 / f if f else 1.0
        self._update_view()
        self._redraw()

    def zoom_by(self, factor, at=None):
        """Zoom by `factor`, keeping the image point under `at` (canvas px; default the centre) where it is."""
        if not self.img_w:
            return
        cx, cy = at if at is not None else (self.cw / 2, self.ch / 2)
        nx = (cx - self.off_x) / self.scale
        ny = (cy - self.off_y) / self.scale
        self.zoom = min(40.0, max(0.25, self.zoom * factor))
        self.pan_x = self.pan_y = 0.0
        self._update_view()
        self.pan_x = cx - (self.off_x + nx * self.scale)
        self.pan_y = cy - (self.off_y + ny * self.scale)
        self._update_view()
        self._redraw()

    def _wheel(self, e):
        self.zoom_by(1.2 if e.delta > 0 else 1 / 1.2, (e.x, e.y))
        return "break"

    def _pan_start(self, e):
        self._pan_from = (e.x, e.y, self.pan_x, self.pan_y)

    def _right_press(self, e):
        if self.task == "segmentation" and self.poly_points:
            return self.finish_polygon()
        self._pan_start(e)

    def _pan_move(self, e):
        if self._pan_from is None:
            return
        x0, y0, px, py = self._pan_from
        self.pan_x, self.pan_y = px + e.x - x0, py + e.y - y0
        self._update_view()
        self._redraw()

    def n2c(self, nx, ny):
        return self.off_x + nx * self.img_w * self.scale, self.off_y + ny * self.img_h * self.scale

    def c2n(self, cx, cy):
        nx = (cx - self.off_x) / max(1e-6, self.img_w * self.scale)
        ny = (cy - self.off_y) / max(1e-6, self.img_h * self.scale)
        return min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))

    def _clamp_canvas(self, cx, cy):
        x0, y0 = self.off_x, self.off_y
        x1, y1 = self.off_x + self.img_w * self.scale, self.off_y + self.img_h * self.scale
        return min(max(cx, x0), x1), min(max(cy, y0), y1)

    def _entry(self):
        """Get-or-create the current image's annotation entry, in-memory only."""
        return self.data["images"].setdefault(self.rel, {"reviewed": False, "boxes": [], "polygons": []})

    # ================================================================== drawing
    def _rect(self, b):
        x0, y0 = self.n2c(b["x"] - b["w"] / 2, b["y"] - b["h"] / 2)
        x1, y1 = self.n2c(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
        return x0, y0, x1, y1

    def _redraw(self):
        self.canvas.delete("all")
        if self.img_bgr is None:
            return
        # Only the visible part of the image is resized: zooming in on a 4K frame stays fast.
        s = self.scale
        ix0, iy0 = max(0, int((0 - self.off_x) / s)), max(0, int((0 - self.off_y) / s))
        ix1 = min(self.img_w, int((self.cw - self.off_x) / s) + 2)
        iy1 = min(self.img_h, int((self.ch - self.off_y) / s) + 2)
        if ix1 > ix0 and iy1 > iy0:
            crop = self.img_rgb[iy0:iy1, ix0:ix1]
            dw, dh = max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(crop, (dw, dh), interpolation=interp)))
            self.canvas.create_image(self.off_x + ix0 * s, self.off_y + iy0 * s, anchor="nw", image=self.photo)

        entry = self.data["images"].get(self.rel, {}) if self.data else {}
        self._show_prediction(entry)
        labels, confs = self.show["labels"].get(), self.show["conf"].get()

        def tag(x0, y0, text, col):
            if labels:
                self.canvas.create_text(x0 + 3, max(6, y0 - 7), text=text, fill=col, anchor="w",
                                        font=(theme.UI, 9, "bold"))

        if self.show["rej"].get():
            for i, p in enumerate(entry.get("rejected") or []):
                sel = self.selected == ("rej", i)
                x0, y0, x1, y1 = self._rect(p)
                col = ACC if sel else COL["rej"]
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=2 if sel else 1, dash=(2, 4))
                tag(x0, y0, f"x {p['cls']}" + (f" {p.get('conf', 0):.2f}" if confs else ""), col)
        if self.show["box"].get():
            for i, box in enumerate(entry.get("boxes") or []):
                sel = self.selected == ("box", i)
                x0, y0, x1, y1 = self._rect(box)
                bad = self._box_problem(box)
                col = ACC if sel else (BAD if bad else COL["corrected"] if box.get("corrected") else COL["box"])
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=3 if sel else 2)
                mark = "! " if bad else ("~ " if box.get("corrected") else "")
                tag(x0, y0, mark + box["cls"], col)
                if sel:
                    for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                        self.canvas.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill=col,
                                                     outline="white")
        if self.show["prop"].get():
            for i, p in enumerate(entry.get("proposals") or []):
                sel = self.selected == ("prop", i)
                x0, y0, x1, y1 = self._rect(p)
                col = ACC if sel else COL["prop"]
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=3 if sel else 2, dash=(6, 3))
                tag(x0, y0, f"? {p['cls']}" + (f" {p.get('conf', 0):.2f}" if confs else ""), col)

        for i, poly in enumerate(entry.get("polygons", [])):
            sel = self.selected == ("poly", i)
            pts = poly["points"]
            flat = []
            for j in range(0, len(pts), 2):
                flat += list(self.n2c(pts[j], pts[j + 1]))
            col = ACC if sel else COL["poly"]
            self.canvas.create_polygon(*flat, outline=col, fill="", width=2 if sel else 1)
            tag(flat[0], flat[1], poly["cls"], col)

        if self.poly_points:
            flat = []
            for j in range(0, len(self.poly_points), 2):
                flat += list(self.n2c(self.poly_points[j], self.poly_points[j + 1]))
            if len(flat) >= 4:
                self.canvas.create_line(*flat, fill=WARN, width=2, dash=(4, 2))
            for j in range(0, len(flat), 2):
                self.canvas.create_oval(flat[j] - 3, flat[j + 1] - 3, flat[j] + 3, flat[j + 1] + 3, fill=WARN,
                                        outline=WARN)

        if self.drag and self.drag.get("mode") == "new":
            x0, y0 = self.drag["start"]
            x1, y1 = self.drag.get("cur", (x0, y0))
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACC, width=2, dash=(3, 2))
        self._update_inspector()

    def _box_problem(self, b) -> str:
        try:
            A._validate_box({k: b[k] for k in ("cls", "x", "y", "w", "h") if k in b}, self.data["classes"])
        except A.AnnotationError as e:
            return str(e)
        x0, y0, x1, y1 = b["x"] - b["w"] / 2, b["y"] - b["h"] / 2, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        if x0 < -1e-6 or y0 < -1e-6 or x1 > 1 + 1e-6 or y1 > 1 + 1e-6:
            return "box extends outside the image"
        if b["w"] < MIN_BOX or b["h"] < MIN_BOX:
            return "box is very small"
        return ""

    def _update_inspector(self):
        entry = self.data["images"].get(self.rel, {}) if self.data and self.rel else {}
        sel = self.selected
        item, kind = None, ""
        if sel:
            lst = {"box": entry.get("boxes"), "prop": entry.get("proposals"), "rej": entry.get("rejected"),
                   "poly": entry.get("polygons")}.get(sel[0]) or []
            if 0 <= sel[1] < len(lst):
                item = lst[sel[1]]
        is_prop = bool(item) and sel[0] == "prop"
        for b in (self.accept_sel_btn, self.reject_sel_btn, self.edit_btn):
            b.configure(state="normal" if is_prop else "disabled")
        self.undo_btn.configure(state="normal" if self.undo_stack else "disabled")
        self.skip_btn.configure(state="normal" if self.rel else "disabled")
        if not item:
            n = len(entry.get("proposals") or [])
            self.insp_state.configure(text="nothing selected", text_color=DIM)
            self.insp_lbl.configure(text=(f"{n} pending proposal(s) on this image. Click one to inspect it, or use "
                                          "Accept all / Reject all." if n else "Click a box to inspect it. Drag on "
                                          "an empty part of the image to draw a new box."))
            self.warn_lbl.configure(text="")
            return
        if sel[0] == "poly":
            self.insp_state.configure(text="POLYGON", text_color=GOOD)
            self.insp_lbl.configure(text=f"class {item['cls']}\n{len(item['points']) // 2} points")
            self.warn_lbl.configure(text="")
            return
        if sel[0] == "prop":
            kind, colour = "PENDING PROPOSAL", WARN
        elif sel[0] == "rej":
            kind, colour = "REJECTED PROPOSAL", BAD
        elif item.get("corrected"):
            kind, colour = "CORRECTED (accepted)", INFO
        elif item.get("source") == "auto":
            kind, colour = "ACCEPTED", GOOD
        else:
            kind, colour = "DRAWN BY HAND", GOOD
        W, H = self.img_w, self.img_h
        x0, y0 = (item["x"] - item["w"] / 2) * W, (item["y"] - item["h"] / 2) * H
        x1, y1 = (item["x"] + item["w"] / 2) * W, (item["y"] + item["h"] / 2) * H
        lines = [f"class       {item['cls']}",
                 f"confidence  {item['conf']:.3f}" if "conf" in item else "confidence  - (no model)",
                 f"model       {item.get('model') or '-'}",
                 f"box px      {x0:.0f}, {y0:.0f} - {x1:.0f}, {y1:.0f}",
                 f"size px     {x1 - x0:.0f} x {y1 - y0:.0f}",
                 f"image       {self.rel}"]
        if item.get("decided"):
            lines.append(f"decided     {item['decided']}")
        if item.get("proposed") and item.get("corrected"):
            o = item["proposed"]
            lines.append(f"model said  {o.get('cls')} {o.get('conf', 0):.2f}")
        self.insp_state.configure(text=kind, text_color=colour)
        self.insp_lbl.configure(text="\n".join(lines), font=(theme.MONO_FACE, 10))
        self.warn_lbl.configure(text=self._box_problem(item))

    # ================================================================== hit-tests
    def _boxes(self):
        return self._entry()["boxes"] if self.rel else []

    def _hit_box(self, cx, cy):
        for i, box in reversed(list(enumerate(self._boxes()))):
            x0, y0, x1, y1 = self._rect(box)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return i
        return None

    def _hit_any(self, cx, cy):
        """The smallest visible item under the pointer (a cap inside a bottle box is picked, not the bottle)."""
        entry = self._entry()
        cands = []
        for kind, key, flag in (("prop", "proposals", "prop"), ("box", "boxes", "box"), ("rej", "rejected", "rej")):
            if not self.show[flag].get():
                continue
            for i, b in enumerate(entry.get(key) or []):
                x0, y0, x1, y1 = self._rect(b)
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    cands.append(((x1 - x0) * (y1 - y0), kind, i))
        return min(cands)[1:] if cands else None

    def _hit_handle(self, cx, cy):
        """Which corner handle of the selected box is under the pointer: 'tl' | 'tr' | 'bl' | 'br' | None."""
        if not self.selected or self.selected[0] != "box":
            return None
        boxes = self._boxes()
        if not 0 <= self.selected[1] < len(boxes):
            return None
        x0, y0, x1, y1 = self._rect(boxes[self.selected[1]])
        for name, hx, hy in (("tl", x0, y0), ("tr", x1, y0), ("bl", x0, y1), ("br", x1, y1)):
            if abs(cx - hx) <= HANDLE + 3 and abs(cy - hy) <= HANDLE + 3:
                return name
        return None

    def _hit_poly(self, cx, cy):
        """Ray-casting point-in-polygon test, in canvas pixel space."""
        polys = self._entry()["polygons"] if self.rel else []
        for i, poly in enumerate(polys):
            pts = poly["points"]
            verts = [self.n2c(pts[j], pts[j + 1]) for j in range(0, len(pts), 2)]
            inside = False
            n = len(verts)
            for a in range(n):
                x1, y1 = verts[a]
                x2, y2 = verts[(a + 1) % n]
                if (y1 > cy) != (y2 > cy):
                    xin = x1 + (cy - y1) / (y2 - y1) * (x2 - x1)
                    if cx < xin:
                        inside = not inside
            if inside:
                return i
        return None

    # ================================================================== mouse editing
    def on_press(self, e):
        if self.img_bgr is None or self.data is None:
            return
        self.canvas.focus_set()
        cx, cy = self._clamp_canvas(e.x, e.y)
        if self.task == "segmentation":
            if not self.poly_points:
                hit = self._hit_poly(cx, cy)
                if hit is not None:
                    self.selected = ("poly", hit)
                    self._redraw()
                    return
            nx, ny = self.c2n(cx, cy)
            self.poly_points += [nx, ny]
            self.selected = None
            self._redraw()
            return
        corner = self._hit_handle(cx, cy)
        if corner:
            box = dict(self._boxes()[self.selected[1]])
            self.drag = {"mode": "resize", "corner": corner, "orig": box, "before": copy.deepcopy(self._entry())}
            return
        hit = self._hit_any(cx, cy)
        if hit is not None:
            self.selected = hit
            if hit[0] == "box":
                nx, ny = self.c2n(cx, cy)
                self.drag = {"mode": "move", "orig": dict(self._boxes()[hit[1]]), "grab": (nx, ny),
                             "before": copy.deepcopy(self._entry())}
            else:
                self.drag = None
            self._redraw()
            return
        self.selected = None
        self.drag = {"mode": "new", "start": (cx, cy), "cur": (cx, cy)}
        self._redraw()

    def on_drag(self, e):
        if self.drag is None or self.img_bgr is None or self.task != "detection":
            return
        cx, cy = self._clamp_canvas(e.x, e.y)
        mode = self.drag["mode"]
        if mode == "new":
            self.drag["cur"] = (cx, cy)
            self._redraw()
        elif mode == "move":
            nx, ny = self.c2n(cx, cy)
            gnx, gny = self.drag["grab"]
            box = dict(self.drag["orig"])
            box["x"] = min(1.0 - box["w"] / 2, max(box["w"] / 2, box["x"] + (nx - gnx)))
            box["y"] = min(1.0 - box["h"] / 2, max(box["h"] / 2, box["y"] + (ny - gny)))
            self._boxes()[self.selected[1]] = box
            self.drag["moved"] = True
            self._redraw()
        elif mode == "resize":
            nx, ny = self.c2n(cx, cy)
            o = self.drag["orig"]
            x0, y0, x1, y1 = o["x"] - o["w"] / 2, o["y"] - o["h"] / 2, o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
            c = self.drag["corner"]
            if "l" in c:
                x0 = min(nx, x1 - MIN_BOX)
            else:
                x1 = max(nx, x0 + MIN_BOX)
            if "t" in c:
                y0 = min(ny, y1 - MIN_BOX)
            else:
                y1 = max(ny, y0 + MIN_BOX)
            box = dict(o)
            box.update(x=(x0 + x1) / 2, y=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0)
            self._boxes()[self.selected[1]] = box
            self.drag["moved"] = True
            self._redraw()

    def on_release(self, e):
        if self.drag is None or self.task != "detection":
            self.drag = None
            self._pan_from = None
            return
        d, self.drag = self.drag, None
        if d["mode"] == "new":
            x0c, y0c = d["start"]
            x1c, y1c = d.get("cur", (x0c, y0c))
            nx0, ny0 = self.c2n(min(x0c, x1c), min(y0c, y1c))
            nx1, ny1 = self.c2n(max(x0c, x1c), max(y0c, y1c))
            w, h = nx1 - nx0, ny1 - ny0
            if w >= MIN_BOX and h >= MIN_BOX:
                self._push_undo("draw box", [self.rel])
                box = {"cls": self.class_menu.get(), "x": nx0 + w / 2, "y": ny0 + h / 2, "w": w, "h": h}
                self._boxes().append(box)
                self.selected = ("box", len(self._boxes()) - 1)
                self._commit("draw_box", [self.rel], cls=box["cls"])
        elif d.get("moved"):
            self.undo_stack.append(("move / resize box", {self.rel: d["before"]}))
            del self.undo_stack[:-UNDO_MAX]
            box = self._boxes()[self.selected[1]]
            corrected = AA.note_correction(box)
            self._commit("edit_box", [self.rel], cls=box["cls"], corrected=corrected)
        self._redraw()

    def finish_polygon(self):
        if self.task != "segmentation":
            return
        if len(self.poly_points) < 6:
            messagebox.showinfo("Too few points", "A polygon needs at least 3 points.")
            return
        entry = self._entry()
        entry["polygons"].append({"cls": self.class_menu.get(), "points": list(self.poly_points)})
        self.poly_points = []
        self.dirty = True
        self._update_progress_label()
        self._redraw()

    def cancel_polygon(self):
        self.poly_points = []
        self._redraw()

    def delete_selected(self):
        if not self.selected or not self.rel:
            return
        kind, i = self.selected
        entry = self._entry()
        if kind == "prop":
            return self.reject_selected()
        lst = {"box": entry.get("boxes"), "poly": entry.get("polygons"), "rej": None}.get(kind)
        if lst is None:
            return
        if 0 <= i < len(lst):
            self._push_undo("delete", [self.rel])
            gone = lst.pop(i)
            self.selected = None
            self._commit("delete_" + kind, [self.rel], cls=gone.get("cls"))
        self.selected = None
        self._redraw()

    # ================================================================== review actions (save at once)
    def _push_undo(self, label, rels):
        snap = {r: copy.deepcopy(self.data["images"].get(r)) for r in rels}
        self.undo_stack.append((label, snap))
        del self.undo_stack[:-UNDO_MAX]

    def undo(self):
        if not self.undo_stack or self.data is None:
            return
        label, snap = self.undo_stack.pop()
        for r, e in snap.items():
            if e is None:
                self.data["images"].pop(r, None)
            else:
                self.data["images"][r] = e
        self.selected = None
        self._commit("undo", list(snap), undone=label, push=False)
        self.status.configure(text=f"undone: {label}", text_color=DIM)

    def _log_path(self) -> Path:
        return self.ann_path.with_name("annotation_log.jsonl")

    def _commit(self, action, rels, push=False, **detail):
        """After a review action: save atomically (validated), write the audit line, refresh what changed."""
        ok = True
        if self.autosave and self.ann_path is not None:
            try:
                A.save_annotations(self.ann_path, self.data)
                self.dirty = False
            except A.AnnotationError as e:
                ok = False
                self.dirty = True
                messagebox.showerror("Cannot save", str(e))
        else:
            self.dirty = True
        if ok:
            AA.log_action(self._log_path(), action, images=rels[:20], n_images=len(rels), **detail)
        for r in rels:
            self._update_row(r)
        self._update_metrics()
        self._update_image_header()
        self._redraw()
        return ok

    def _sel_prop(self):
        if self.selected and self.selected[0] == "prop" and self.rel:
            n = len(self._entry().get("proposals") or [])
            if 0 <= self.selected[1] < n:
                return self.selected[1]
        return None

    def accept_selected(self):
        i = self._sel_prop()
        if i is None:
            return
        self._push_undo("accept", [self.rel])
        p = self._entry()["proposals"][i]
        AA.accept(self.data, self.rel, [i])
        self.selected = ("box", len(self._boxes()) - 1)
        self._commit("accept", [self.rel], cls=p["cls"], conf=p.get("conf"), model=p.get("model"))

    def reject_selected(self):
        i = self._sel_prop()
        if i is None:
            return
        self._push_undo("reject", [self.rel])
        p = self._entry()["proposals"][i]
        AA.reject(self.data, self.rel, [i])
        self.selected = None
        self._commit("reject", [self.rel], cls=p["cls"], conf=p.get("conf"), model=p.get("model"))

    def edit_selected(self):
        i = self._sel_prop()
        if i is None:
            return
        self._push_undo("edit", [self.rel])
        j = AA.edit_proposal(self.data, self.rel, i)
        self.selected = ("box", j)
        self.class_menu.set(self._boxes()[j]["cls"])
        self._commit("edit_start", [self.rel], cls=self._boxes()[j]["cls"])
        self.status.configure(text="Editing: drag the box or its corners, or pick a class and press Apply. "
                                   "Any change is recorded as CORRECTED.", text_color=DIM)

    def apply_class(self):
        if not self.selected or not self.rel or self.data is None:
            return
        cls = self.class_menu.get()
        kind, i = self.selected
        if kind == "prop":
            self._push_undo("change class", [self.rel])
            i = AA.edit_proposal(self.data, self.rel, i)
            self.selected = ("box", i)
        elif kind == "box":
            self._push_undo("change class", [self.rel])
        else:
            return
        box = self._boxes()[i]
        box["cls"] = cls
        corrected = AA.note_correction(box)
        self._commit("change_class", [self.rel], cls=cls, corrected=corrected)

    def skip_image(self):
        if not self.rel or self.data is None:
            return
        self._push_undo("skip", [self.rel])
        AA.skip(self.data, self.rel)
        self._commit("skip", [self.rel])
        self.goto_next_pending()

    def accept_proposals(self):
        """Accept every pending proposal on this image."""
        if not self.rel or not self._entry().get("proposals"):
            return
        self._push_undo("accept all", [self.rel])
        n = AA.accept(self.data, self.rel)
        self.selected = None
        self._commit("accept_all", [self.rel], n=n)

    def reject_proposals(self):
        if not self.rel or not self._entry().get("proposals"):
            return
        self._push_undo("reject all", [self.rel])
        n = AA.reject(self.data, self.rel)
        self.selected = None
        self._commit("reject_all", [self.rel], n=n)

    def mark_reviewed(self):
        if not self.rel or self.data is None:
            return
        n = len(self._entry().get("proposals") or [])
        if n and not messagebox.askyesno(
                "Undecided proposals", f"{n} proposal(s) on this image are still undecided.\n\n"
                                       "Reject them and mark the image reviewed?"):
            return
        self._push_undo("mark reviewed", [self.rel])
        rejected = AA.mark_reviewed(self.data, self.rel, reject_pending=bool(n))
        self._commit("mark_reviewed", [self.rel], rejected=rejected)
        self._highlight_row(self.rel)

    # ================================================================== bulk
    def bulk_dialog(self, action: str):
        if self.data is None:
            return
        BulkDialog(self, action)

    def run_bulk(self, action: str, selection: dict, scope_note: str = "") -> int:
        """Apply a confirmed bulk selection (only pending proposals). One undo step, one audit line."""
        if not selection:
            return 0
        self._push_undo(f"bulk {action}", list(selection))
        n = AA.bulk(self.data, selection, action)
        self.selected = None
        self._commit("bulk_" + action, list(selection), n=n, scope=scope_note)
        self.apply_filter()
        return n

    # ================================================================== keyboard
    @staticmethod
    def _typing(widget) -> bool:
        try:
            cls = widget.winfo_class()
        except Exception:                                        # noqa: BLE001
            return False
        return isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox)) or cls in ("Entry", "Text", "TEntry",
                                                                               "TCombobox", "Spinbox")

    def _active(self) -> bool:
        try:
            if not self.body.winfo_ismapped():
                return False
        except Exception:                                        # noqa: BLE001
            return False
        tabs = getattr(self.app, "tabs", None)
        return tabs is None or tabs.get() == "Annotate"

    def _on_key(self, e):
        """Review shortcuts. Ignored while typing in a text field and when this page is not on screen."""
        if self.data is None or self._typing(e.widget) or not self._active():
            return None
        k = (e.keysym or "").lower()
        ctrl = bool(e.state & 0x4)
        if ctrl and k == "z":
            self.undo()
            return "break"
        if ctrl:
            return None
        act = {"a": self.accept_selected, "r": self.reject_selected, "e": self.edit_selected, "s": self.skip_image,
               "n": self.next_image, "p": self.prev_image, "f": self.fit, "plus": lambda: self.zoom_by(1.25),
               "equal": lambda: self.zoom_by(1.25), "kp_add": lambda: self.zoom_by(1.25),
               "minus": lambda: self.zoom_by(1 / 1.25), "kp_subtract": lambda: self.zoom_by(1 / 1.25),
               "escape": self._escape, "delete": self.delete_selected}.get(k)
        if act is None:
            return None
        act()
        return "break"

    def _escape(self):
        self.selected, self.drag, self.poly_points = None, None, []
        self._redraw()

    # ================================================================== proposal generation (background)
    def _line_running(self) -> bool:
        pt = getattr(self.app, "tab_production", None)
        return getattr(pt, "mstate", "") in ("RUNNING", "INSPECTING", "INITIALIZING", "STOPPING")

    def _scope_images(self) -> list:
        scope = self.scope_menu.get()
        if scope.startswith("Failed"):
            return [r for r in self.images if (self.data["images"].get(r) or {}).get("failed")
                    and not (self.data["images"].get(r) or {}).get("reviewed")]
        if scope.startswith("Current list"):
            return [r for r in self.view if not (self.data["images"].get(r) or {}).get("reviewed")]
        out = []
        for r in self.images:
            e = self.data["images"].get(r) or {}
            if not (e.get("reviewed") or e.get("boxes") or e.get("polygons") or e.get("proposals") or e.get("auto")
                    or e.get("failed")):
                out.append(r)
        return out

    def preflight(self) -> tuple:
        """(images, weights path, min_conf, problems) -- checked BEFORE anything runs."""
        problems = []
        if self.data is None or self.task != "detection":
            return [], None, 0.25, ["Open a detection dataset first."]
        if self.job is not None:
            problems.append("A proposal run is already in progress.")
        if self._line_running():
            problems.append("The inspection line is running: generate proposals when it is stopped "
                            "(one GPU job at a time).")
        label = self.model_menu.get()
        path = dict(self._models).get(label) if getattr(self, "_models", None) else None
        if path is None or not Path(path).is_file():
            problems.append("No detector weights selected (or the file is missing). Nothing is substituted.")
        try:
            min_conf = float(self.conf_entry.get())
            if not 0.0 < min_conf < 1.0:
                raise ValueError
        except ValueError:
            problems.append("Min conf must be a number between 0 and 1.")
            min_conf = 0.25
        images = self._scope_images()
        if not images:
            problems.append("No image in the chosen scope (reviewed images are never re-proposed).")
        try:
            t = self.ann_path.with_name(".write_test")
            t.write_text("x")
            t.unlink()
        except OSError as e:
            problems.append(f"Cannot write next to {self.ann_path.name}: {e}")
        return images, path, min_conf, problems

    def auto_annotate(self):
        images, path, min_conf, problems = self.preflight()
        if problems:
            return messagebox.showinfo("Generate proposals", "\n".join(problems))
        replace = self.scope_menu.get().startswith("Current list")
        if not messagebox.askyesno(
                "Generate proposals",
                f"Run {Path(path).name} on {len(images)} image(s), min confidence {min_conf:.2f}?\n\n"
                + ("Their current PENDING proposals will be replaced (accepted / rejected boxes are kept).\n\n"
                   if replace else "")
                + "Results are proposals: nothing becomes a label until you accept it."):
            return
        self.start_job(images, path, min_conf)

    def start_job(self, images, weights, min_conf, detector_factory=None):
        """Start the one background job. detector_factory(weights, min_conf) -> (detect_fn, class_names, model_id)
        (the self-test injects a fake; the default loads the YOLO checkpoint on the worker thread)."""
        job = {"ann_path": self.ann_path, "images": list(images), "weights": weights, "min_conf": min_conf,
               "cancel": threading.Event(), "running": threading.Event(), "done": 0, "stats": {}, "t0": time.time(),
               "model_id": None, "error": None, "finished": False}
        job["running"].set()
        self.job = job
        self._error = ""
        self.pill.set("GENERATING")
        self.auto_btn.configure(state="disabled")
        self.prog_bar.set(0)
        self.prog_lbl.configure(text=f"loading model {Path(weights).name} ...")
        classes = list(self.data["classes"])
        root = self.image_root if self.image_root is not None else D.IMAGE_ROOT
        AA.log_action(self._log_path(), "generate_start", model=str(weights), n_images=len(images), min_conf=min_conf)

        def factory(w, conf):
            from vision import detect
            det = detect.YoloDetector(weights=w, conf=conf, verify=Path(w) == detect.DEFAULT_WEIGHTS, require=())
            return AA.yolo_detect_fn(det), list(det.class_names), det.model_id

        def work():
            try:
                fn, names, model_id = (detector_factory or factory)(weights, min_conf)
            except Exception as e:                               # noqa: BLE001 - shown, never swallowed
                msg = f"cannot load the model: {type(e).__name__}: {e}"   # bound now: `e` is deleted after except
                return self._post(lambda m=msg: self._job_failed(job, m))
            mapped, unmapped, never = AA.class_mapping(names, classes)
            if not mapped:
                msg = f"incompatible model: its classes {names} share none with this dataset's {classes}"
                return self._post(lambda m=msg: self._job_failed(job, m))
            job["model_id"] = model_id
            self._post(lambda: self._job_info(job, mapped, unmapped, never))
            batch, last = [], time.time()
            for res in AA.detect_images(job["images"], root, fn, D.imread, should_stop=job["cancel"].is_set,
                                        wait=job["running"].wait):
                if res[0] == "aborted":
                    job["error"] = res[3]
                    break
                batch.append(res)
                if len(batch) >= BATCH or time.time() - last > 1.0:
                    b, batch = batch, []
                    self._post(lambda b=b: self._apply_batch(job, b))
                    last = time.time()
            if batch:
                self._post(lambda b=batch: self._apply_batch(job, b))
            self._post(lambda: self._job_done(job))

        threading.Thread(target=work, daemon=True, name="annotate-generate").start()

    def _job_info(self, job, mapped, unmapped, never):
        msg = f"model {job['model_id']}: proposes {', '.join(mapped)}"
        if unmapped:
            msg += f"  |  not in this dataset (ignored): {', '.join(unmapped)}"
        if never:
            msg += f"  |  never proposed by this model: {', '.join(never)}"
        self.status.configure(text=msg, text_color=DIM)

    def _apply_batch(self, job, batch):
        """Merge results into the job's OWN annotations file. If the page now shows another dataset (project
        switched), the file is loaded, merged and saved on disk: proposals never land in the wrong project."""
        same = self.data is not None and self.ann_path == job["ann_path"]
        target = self.data if same else A.load_annotations(job["ann_path"])
        for res in batch:
            AA.apply_result(target, res, job["min_conf"], job["model_id"], job["stats"],
                            overwrite=False)
        try:
            A.save_annotations(job["ann_path"], target)
        except A.AnnotationError as e:
            job["error"] = f"save failed: {e}"
            job["cancel"].set()
            job["running"].set()
        job["done"] += len(batch)
        if same:
            for res in batch:
                self._update_row(res[1])
            self._update_metrics()
            if self.rel in {r[1] for r in batch}:
                self._redraw()
        el = max(1e-6, time.time() - job["t0"])
        st = job["stats"]
        n = len(job["images"])
        rate = job["done"] / el
        eta = (n - job["done"]) / rate if rate > 0 else 0
        self.prog_bar.set(job["done"] / max(1, n))
        self.prog_lbl.configure(text=f"{job['done']}/{n} images  ·  {rate:.1f} img/s  ·  {st.get('proposed', 0)} "
                                     f"proposals  ·  {st.get('empty', 0)} empty  ·  {st.get('failed', 0)} failed  ·  "
                                     f"ETA {eta / 60:.1f} min")

    def _job_failed(self, job, msg):
        job["error"] = msg
        job["finished"] = True
        self.job = None
        self._error = msg
        self.pill.set("ERROR")
        self.auto_btn.configure(state="normal")
        self.prog_lbl.configure(text="")
        self.status.configure(text=msg, text_color=BAD)
        AA.log_action(self._log_path() if self.ann_path else job["ann_path"].with_name("annotation_log.jsonl"),
                      "generate_error", error=msg)
        if self.app is not None:
            messagebox.showerror("Generate proposals", msg)

    def _job_done(self, job):
        job["finished"] = True
        self.job = None
        st = job["stats"]
        cancelled = job["cancel"].is_set() and not job["error"]
        words = (f"{'CANCELLED' if cancelled else 'STOPPED' if job['error'] else 'DONE'}: {st.get('images', 0)} "
                 f"images proposed ({st.get('proposed', 0)} proposals, {st.get('empty', 0)} with no object), "
                 f"{st.get('failed', 0)} failed, {st.get('invalid', 0)} invalid detections dropped, "
                 f"{st.get('skipped', 0)} skipped (reviewed meanwhile).")
        if job["error"]:
            words += f"  {job['error']}"
        self.status.configure(text=words + "  Completed work is saved.", text_color=BAD if job["error"] else DIM)
        self.prog_lbl.configure(text=words[:140])
        self.auto_btn.configure(state="normal" if self.task == "detection" else "disabled")
        self.pause_btn.configure(text="Pause")
        AA.log_action(job["ann_path"].with_name("annotation_log.jsonl"), "generate_end", model=job["model_id"],
                      cancelled=cancelled, error=job["error"], **{k: v for k, v in st.items()})
        self.pill.set("ERROR" if job["error"] else "IDLE")
        if not job["error"]:
            self._update_metrics()
        if self.data is not None and self.ann_path == job["ann_path"]:
            self.apply_filter()

    def toggle_pause(self):
        j = self.job
        if j is None:
            return
        if j["running"].is_set():
            j["running"].clear()
            self.pill.set("PAUSED")
            self.pause_btn.configure(text="Resume")
        else:
            j["running"].set()
            self.pill.set("GENERATING")
            self.pause_btn.configure(text="Pause")

    def cancel_job(self):
        j = self.job
        if j is None:
            return
        j["cancel"].set()
        j["running"].set()                       # a paused job must wake up to see the cancel
        self.pill.set("CANCELLING")

    # ================================================================== save / load / history
    def save(self):
        if self.data is None:
            return
        try:
            A.save_annotations(self.ann_path, self.data)
        except A.AnnotationError as e:
            messagebox.showerror("Cannot save", str(e))
            return
        self.dirty = False
        self.status.configure(text=f"Saved {self.ann_path.name}", text_color=DIM)
        AA.log_action(self._log_path(), "save")
        self._rebuild_image_list()
        self._update_metrics()

    def reload_annotations(self):
        if self.ann_path is None:
            return
        if self.dirty and not messagebox.askyesno("Reload from disk",
                                                  "Discard unsaved edits and reload annotations.json?"):
            return
        try:
            self.data = A.load_annotations(self.ann_path)
        except A.AnnotationError as e:
            messagebox.showerror("Cannot reload", str(e))
            return
        self.dirty = False
        self.undo_stack.clear()
        self._populate_classes()
        self.apply_filter()
        self._update_metrics()
        if self.rel:
            self.load_image(self.rel)

    def show_history(self):
        if self.ann_path is None:
            return
        recs = AA.read_log(self._log_path(), 300)
        win = ctk.CTkToplevel(self.parent)
        win.title("Annotation history")
        win.geometry("760x480")
        box = ctk.CTkTextbox(win, font=(theme.MONO_FACE, 11))
        box.pack(fill="both", expand=True, padx=8, pady=8)
        lines = [f"{r.get('time')}  {r.get('user', ''):<10} {r.get('action'):<16} "
                 + " ".join(f"{k}={v}" for k, v in r.items() if k not in ("time", "user", "action", "images"))
                 for r in reversed(recs)]
        box.insert("end", "\n".join(lines) or "no actions recorded yet")
        box.configure(state="disabled")

    # ================================================================== export
    def export_detection(self):
        if self.task != "detection" or self.data is None:
            return
        s = A.export_summary(self.data, "boxes")
        if not s["exported"]:
            return messagebox.showinfo("Export approved annotations",
                                       f"Nothing to export yet: no image has accepted boxes or is marked reviewed.\n"
                                       f"{s['skipped_pending']} image(s) still hold undecided proposals.")
        objs = ", ".join(f"{c} {n}" for c, n in s["objects"].items() if n) or "none"
        if not messagebox.askyesno(
                "Export approved annotations",
                f"Export {s['exported']} image label file(s) ({s['background']} reviewed background images).\n"
                f"Objects: {objs}.\n\nLeft out: {s['skipped_pending']} image(s) with undecided proposals, "
                f"{s['skipped_unannotated']} not annotated.\nPending and rejected proposals are never exported."):
            return
        out = filedialog.askdirectory(title="Export YOLO detection labels to...")
        if not out:
            return
        self._export(out, A.export_yolo_detection, "boxes")

    def export_segmentation(self):
        if self.task != "segmentation" or self.data is None:
            return
        out = filedialog.askdirectory(title="Export YOLO segmentation labels to...")
        if not out:
            return
        self._export(out, A.export_yolo_segmentation, "polygons")

    def _export(self, out, fn, key):
        self.pill.set("EXPORTING")
        self.parent.update_idletasks()
        try:
            written = fn(self.data, Path(out))
            man = A.write_export_manifest(Path(out), self.data, written, self.ann_path, key)
        except A.AnnotationError as e:
            self.pill.set("ERROR")
            messagebox.showerror("Export failed", str(e))
            return
        AA.log_action(self._log_path(), "export", out=str(out), files=len(written), key=key)
        self._update_metrics()
        messagebox.showinfo("Exported", f"Wrote {len(written)} label file(s) to {out}\nRecord: {man.name}")


# ====================================================================== bulk dialog
class BulkDialog(ctk.CTkToplevel):
    """Bulk accept / reject of PENDING proposals by class, confidence range, model and scope. Shows the exact count
    before anything changes and asks once more; one undo step."""

    def __init__(self, tab: AnnotationTab, action: str):
        super().__init__(tab.parent)
        self.tab, self.action = tab, action
        self.title(f"Bulk {action}")
        self.geometry("460x330")
        self.transient(tab.parent.winfo_toplevel())
        C.section(self, f"Bulk {action} - pending proposals only")
        g = ctk.CTkFrame(self, fg_color="transparent")
        g.pack(fill="x", padx=14)
        ctk.CTkLabel(g, text="Class", width=110, anchor="w").grid(row=0, column=0, sticky="w", pady=3)
        self.cls = ctk.CTkOptionMenu(g, values=["all classes"] + list(tab.data["classes"]), command=self._count)
        self.cls.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(g, text="Confidence", width=110, anchor="w").grid(row=1, column=0, sticky="w", pady=3)
        r = ctk.CTkFrame(g, fg_color="transparent")
        r.grid(row=1, column=1, sticky="w")
        self.lo, self.hi = ctk.CTkEntry(r, width=60), ctk.CTkEntry(r, width=60)
        self.lo.insert(0, "0.90" if action == "accept" else "0")
        self.hi.insert(0, "1" if action == "accept" else "0.30")
        self.lo.pack(side="left")
        ctk.CTkLabel(r, text=" to ").pack(side="left")
        self.hi.pack(side="left")
        for e in (self.lo, self.hi):
            e.bind("<KeyRelease>", lambda _e: self._count())
        ctk.CTkLabel(g, text="Model", width=110, anchor="w").grid(row=2, column=0, sticky="w", pady=3)
        self.model = ctk.CTkOptionMenu(g, values=["all models"] + AA.models_used(tab.data), command=self._count)
        self.model.grid(row=2, column=1, sticky="w")
        ctk.CTkLabel(g, text="Images", width=110, anchor="w").grid(row=3, column=0, sticky="w", pady=3)
        self.scope = ctk.CTkOptionMenu(g, values=["current list (filters on the left)", "this image", "all images"],
                                       command=self._count)
        self.scope.grid(row=3, column=1, sticky="w")
        self.count_lbl = ctk.CTkLabel(self, text="", font=(theme.UI, 14, "bold"))
        self.count_lbl.pack(pady=(12, 4))
        ctk.CTkLabel(self, text="Accepted, corrected, rejected and hand-drawn boxes are never changed.",
                     text_color=DIM, font=theme.TINY).pack()
        b = ctk.CTkFrame(self, fg_color="transparent")
        b.pack(pady=10)
        self.go = C.button(b, f"{action.capitalize()} ...", self.apply, "success" if action == "accept" else "danger",
                           140)
        self.go.pack(side="left", padx=4)
        C.button(b, "Close", self.destroy, width=90).pack(side="left", padx=4)
        self._count()

    def selection(self) -> tuple:
        t = self.tab
        sc = self.scope.get()
        imgs = [t.rel] if sc == "this image" and t.rel else t.images if sc == "all images" else t.view
        try:
            lo, hi = float(self.lo.get()), float(self.hi.get())
        except ValueError:
            return {}, sc
        cls = None if self.cls.get() == "all classes" else self.cls.get()
        model = None if self.model.get() == "all models" else self.model.get()
        return AA.select_proposals(t.data, imgs, cls, lo, hi, model), sc

    def _count(self, *_):
        sel, _sc = self.selection()
        n = sum(len(v) for v in sel.values())
        self.count_lbl.configure(text=f"{n} proposal(s) on {len(sel)} image(s) will be {self.action}ed")
        self.go.configure(state="normal" if n else "disabled")

    def apply(self):
        sel, sc = self.selection()
        n = sum(len(v) for v in sel.values())
        if not n or not messagebox.askyesno(f"Bulk {self.action}",
                                            f"{self.action.capitalize()} {n} proposal(s) on {len(sel)} image(s)?\n"
                                            "This is one step: Undo restores all of them."):
            return
        done = self.tab.run_bulk(self.action, sel, f"{sc}; class={self.cls.get()}; conf={self.lo.get()}-"
                                                   f"{self.hi.get()}; model={self.model.get()}")
        self.tab.status.configure(text=f"bulk {self.action}: {done} proposal(s)", text_color=DIM)
        self.destroy()


# ------------------------------------------------------------------ self-test

def selftest():
    """Exercises AnnotationTab against disposable detection / segmentation / classification projects created
    under a temporary projects/ root (the isolation dataset.project_demo() uses), so nothing under the real
    projects/ directory -- in particular projects/om_bottle/ -- is touched. Mouse motion itself is not
    synthesised (unreliable headless); the exact methods the mouse handlers call are driven instead."""
    import tempfile
    import tkinter.messagebox as mb

    import cv2 as _cv2
    import numpy as np

    was = (D.PROJECTS, D.ACTIVE_TXT, D.PROJECT)
    orig = (mb.showerror, mb.showinfo, mb.askyesno)
    popups = []
    mb.showerror = lambda title, msg, *a, **k: popups.append(("error", title, msg))
    mb.showinfo = lambda title, msg, *a, **k: popups.append(("info", title, msg))
    mb.askyesno = lambda *a, **k: True
    blank = np.zeros((60, 40, 3), np.uint8)

    def wait_job(tab, root, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            tab._drain()
            root.update()
            if tab.job is None:
                tab._drain()
                return
            time.sleep(0.02)
        raise AssertionError("generation job did not finish")

    try:
        with tempfile.TemporaryDirectory() as td:
            D.PROJECTS = Path(td) / "projects"
            D.ACTIVE_TXT = D.PROJECTS / "active.txt"

            # ---- detection project -----------------------------------------
            name = D.use_project(D.create_project("Selftest Detection", task="detection"))
            assert D.project_task(name) == "detection"
            img_path = D.IMAGE_ROOT / D.INBOX_DIR / "img001.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            _cv2.imencode(".jpg", blank)[1].tofile(str(img_path))

            root = ctk.CTk()
            root.withdraw()
            tab = AnnotationTab(app=None, parent=ctk.CTkFrame(root))
            tab.refresh()
            assert not tab.body.winfo_manager() == "pack", "body should stay hidden until annotations.json exists"
            entry_widget = None
            for w in tab.init_frame.winfo_children():
                if isinstance(w, ctk.CTkEntry):
                    entry_widget = w
            assert entry_widget is not None, "class-name entry not found in init prompt"
            entry_widget.insert(0, "scratch, dent")
            data = A.init_annotations("detection", ["scratch", "dent"])
            A.save_annotations(D.PROJECT_DIR / "annotations.json", data)
            tab.refresh()
            assert tab.body.winfo_manager() == "pack", "body should show once annotations.json exists"
            assert tab.class_menu.cget("values") == ["scratch", "dent"]
            assert tab.rel == "_inbox/img001.jpg"

            # simulate a completed box drag: same code on_release runs
            tab.class_menu.set("scratch")
            tab._boxes().append({"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3})
            tab.selected = ("box", 0)
            tab.dirty = True
            tab._redraw()
            assert len(tab._boxes()) == 1
            box = dict(tab._boxes()[0])
            box["x"], box["y"] = 0.6, 0.4
            tab._boxes()[0] = box
            assert tab._boxes()[0]["x"] == 0.6 and tab._boxes()[0]["y"] == 0.4

            before = tab.ann_path.read_text(encoding="utf-8")
            tab._boxes().append({"cls": "not_a_class", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1})
            popups.clear()
            tab.save()
            assert tab.ann_path.read_text(encoding="utf-8") == before, \
                "save() must not have written a box with an unknown class to disk"
            assert popups and popups[-1][0] == "error", "save() swallowed the validation failure silently"
            tab._boxes().pop()
            tab.save()
            reloaded = A.load_annotations(tab.ann_path)
            assert reloaded["images"]["_inbox/img001.jpg"]["boxes"][0]["cls"] == "scratch"
            print("ok  detection: init, box create/move, unknown-class save-rejection, save/reload")

            tab.selected = ("box", 0)
            tab.delete_selected()
            assert tab._boxes() == []
            assert A.load_annotations(tab.ann_path)["images"]["_inbox/img001.jpg"]["boxes"] == [], "delete not saved"
            tab.undo()
            assert len(tab._boxes()) == 1 and len(A.load_annotations(tab.ann_path)["images"]["_inbox/img001.jpg"]
                                                  ["boxes"]) == 1, "undo not saved"
            tab.selected = ("box", 0)
            tab.delete_selected()
            print("ok  detection: delete selected box saved at once; undo restores it (saved too)")

            with tempfile.TemporaryDirectory() as outdir:
                tab._boxes().append({"cls": "dent", "x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1})
                tab.save()
                written = A.export_yolo_detection(tab.data, Path(outdir))
                assert written == ["_inbox/img001.jpg"]
                txt = (Path(outdir) / "img001.txt").read_text().strip()
                assert txt == "1 0.300000 0.300000 0.100000 0.100000"
            print("ok  detection: YOLO export via annotate.py")

            # ---- zoom / pan keep the coordinate mapping exact
            tab.cw, tab.ch = 400, 300
            tab.fit()
            p0 = tab.n2c(0.25, 0.75)
            tab.zoom_by(2.0, at=p0)
            p1 = tab.n2c(0.25, 0.75)
            assert abs(p0[0] - p1[0]) < 1e-6 and abs(p0[1] - p1[1]) < 1e-6, "zoom moved the point under the cursor"
            nx, ny = tab.c2n(*tab.n2c(0.4, 0.6))
            assert abs(nx - 0.4) < 1e-9 and abs(ny - 0.6) < 1e-9
            tab.fit()
            print("ok  zoom keeps the point under the cursor; canvas <-> image mapping is exact")

            # ---- generation through the real background path, with a fake detector: progress, failures,
            #      valid empty results, invalid detections, model identity
            for i in range(2, 9):
                _cv2.imencode(".jpg", blank)[1].tofile(str(D.IMAGE_ROOT / D.INBOX_DIR / f"img00{i}.jpg"))
            (D.IMAGE_ROOT / D.INBOX_DIR / "img009.jpg").write_bytes(b"not a jpeg")          # unreadable
            tab.refresh()
            tab.scope_menu.set(SCOPES[0])
            calls = {"n": 0}

            def fake_detect(img):
                calls["n"] += 1
                k = calls["n"]
                if k == 3:
                    return []                                                                  # valid, empty
                if k == 4:
                    raise RuntimeError("inference blew up")
                if k == 5:
                    return [("dent", float("nan"), 1, 1, 10, 10), ("scratch", 0.8, 2, 2, 30, 50)]
                return [("scratch", 0.9, 2, 2, 30, 50), ("dent", 0.4, 5, 5, 20, 20), ("person", 0.99, 0, 0, 5, 5)]
            imgs, w, mc, probs = tab.preflight()
            assert any("weights" in p for p in probs) or w is not None
            todo = tab._scope_images()
            assert "_inbox/img001.jpg" not in todo and "_inbox/img009.jpg" in todo and len(todo) == 8, todo
            tab.start_job(todo, Path(td) / "fake.pt", 0.25,
                          detector_factory=lambda w_, c_: (fake_detect, ["scratch", "dent", "person"], "fake@abc"))
            assert tab.pill.state == "GENERATING"
            wait_job(tab, root)
            on_disk = A.load_annotations(tab.ann_path)["images"]
            st = AA.counts(A.load_annotations(tab.ann_path), tab.images)
            assert st["failed"] == 2, st                                         # unreadable + inference error
            assert "unreadable" in on_disk["_inbox/img009.jpg"]["failed"]["reason"]
            empties = [r for r, e in on_disk.items() if (e.get("auto") or {}).get("n") == 0]
            assert len(empties) == 1, empties
            props = [p for e in on_disk.values() for p in e.get("proposals") or []]
            assert props and all(p["model"] == "fake@abc" for p in props)
            assert not any(p["cls"] == "person" for p in props)                 # a class the dataset lacks: dropped
            assert all(not e.get("boxes") for r, e in on_disk.items() if r != "_inbox/img001.jpg"), "proposal in boxes"
            assert tab.pill.state in ("IDLE", "REVIEWING")
            print("ok  generation (background thread, fake detector): batches saved, unreadable + inference error = "
                  "FAILED, empty result kept, NaN dropped, unknown class ignored, model id recorded")

            # incompatible model / load failure -> ERROR, nothing written
            snap = tab.ann_path.read_text(encoding="utf-8")
            tab.start_job(["_inbox/img002.jpg"], Path(td) / "x.pt", 0.25,
                          detector_factory=lambda w_, c_: (fake_detect, ["person", "car"], "coco@1"))
            wait_job(tab, root)
            assert tab.pill.state == "ERROR" and "incompatible" in tab._error, tab._error
            assert tab.ann_path.read_text(encoding="utf-8") == snap

            def broken(w_, c_):
                raise FileNotFoundError("no such weights")
            tab.start_job(["_inbox/img002.jpg"], Path(td) / "missing.pt", 0.25, detector_factory=broken)
            wait_job(tab, root)
            assert tab.pill.state == "ERROR" and "cannot load" in tab._error
            assert tab.ann_path.read_text(encoding="utf-8") == snap
            print("ok  missing model and incompatible classes -> ERROR before any image, file unchanged")

            # cancel: completed batches stay saved, the rest is untouched, nothing corrupt
            slow_calls = {"n": 0}

            def slow(img):
                slow_calls["n"] += 1
                time.sleep(0.05)
                return [("scratch", 0.7, 2, 2, 30, 50)]
            fresh = [f"_inbox/c{i:02d}.jpg" for i in range(30)]
            for r in fresh:
                _cv2.imencode(".jpg", blank)[1].tofile(str(D.IMAGE_ROOT / r))
            tab.refresh()
            tab.start_job(fresh, Path(td) / "fake.pt", 0.25,
                          detector_factory=lambda w_, c_: (slow, ["scratch"], "slow@1"))
            t0 = time.time()
            while slow_calls["n"] < 10 and time.time() - t0 < 10:
                tab._drain()
                root.update()
                time.sleep(0.01)
            tab.cancel_job()
            wait_job(tab, root)
            disk = A.load_annotations(tab.ann_path)                               # valid file (load validates)
            done = [r for r in fresh if (disk["images"].get(r) or {}).get("auto")]
            assert 0 < len(done) < len(fresh), len(done)
            assert all(disk["images"][r]["proposals"] for r in done)
            assert any(r["action"] == "generate_end" and r.get("cancelled") for r in AA.read_log(tab._log_path()))
            print(f"ok  cancel after {len(done)}/{len(fresh)} images: completed work saved, file valid, logged")

            # ---- per-proposal review: accept / reject / edit -> corrected; export only approved
            tab.refresh()
            rel = next(r for r in tab.images if len((tab.data["images"].get(r) or {}).get("proposals") or []) >= 2)
            tab.goto_image(rel)
            e = tab._entry()
            n0 = len(e["proposals"])
            tab.selected = ("prop", 0)
            tab.accept_selected()
            assert len(tab._entry()["proposals"]) == n0 - 1 and tab._entry()["boxes"][-1]["source"] == "auto"
            tab.selected = ("prop", 0)
            tab.edit_selected()
            j = tab.selected[1]
            was_cls = tab._entry()["boxes"][j]["proposed"]["cls"]
            assert not tab._entry()["boxes"][j]["corrected"]                     # unchanged so far
            new_cls = "dent" if was_cls == "scratch" else "scratch"
            tab.class_menu.set(new_cls)
            tab.apply_class()
            b = tab._entry()["boxes"][j]
            assert b["corrected"] and b["proposed"]["cls"] == was_cls and b["cls"] == new_cls
            # an image still holding a proposal is NOT exported
            tab._entry()["proposals"].append({"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "conf": 0.5})
            tab.save()
            with tempfile.TemporaryDirectory() as outdir:
                assert rel not in A.export_yolo_detection(tab.data, Path(outdir))
            tab.selected = ("prop", len(tab._entry()["proposals"]) - 1)
            tab.reject_selected()
            assert not tab._entry()["proposals"] and tab._entry()["rejected"]
            with tempfile.TemporaryDirectory() as outdir:
                assert rel in A.export_yolo_detection(tab.data, Path(outdir))
                lines = (Path(outdir) / (Path(rel).stem + ".txt")).read_text().split("\n")
                ids = sorted(int(x.split()[0]) for x in lines if x.strip())
                want = sorted(tab.data["classes"].index(bx["cls"]) for bx in tab._entry()["boxes"])
                assert len(ids) == 2 and ids == want, (ids, want)               # accepted + corrected, as edited
                tab._export(outdir, A.export_yolo_detection, "boxes")
                assert (Path(outdir) / "export_manifest.json").is_file()
            tab.undo()                                                           # undo the reject
            assert len(tab._entry()["proposals"]) == 1
            acts = [r["action"] for r in AA.read_log(tab._log_path())]
            for a in ("accept", "edit_start", "change_class", "reject", "undo", "export"):
                assert a in acts, (a, acts)
            print("ok  accept / edit -> corrected (original kept) / reject; pending image not exported; approved "
                  "export = accepted + corrected only, manifest written; undo; every action in the audit log")

            # mark reviewed with undecided proposals: asked; refused when declined
            mb.askyesno = lambda *a, **k: False
            tab.mark_reviewed()
            assert not tab._entry().get("reviewed")
            mb.askyesno = lambda *a, **k: True
            tab.mark_reviewed()
            assert tab._entry()["reviewed"] and not tab._entry()["proposals"]
            print("ok  mark reviewed asks about undecided proposals (declined = nothing changes)")

            # ---- bulk: exact scope, one undo step
            tab.state_menu.set("Pending proposals")
            tab.apply_filter()
            assert all("pending" in AA.image_states(tab.data["images"][r]) for r in tab.view)
            sel = AA.select_proposals(tab.data, tab.view, cls="scratch", lo=0.85, hi=1.0)
            n_sel = sum(len(v) for v in sel.values())
            other_before = {r: copy.deepcopy(tab.data["images"][r]) for r in tab.images
                            if r not in sel and r in tab.data["images"]}
            assert tab.run_bulk("accept", sel) == n_sel > 0
            for r, e0 in other_before.items():
                assert tab.data["images"][r] == e0, f"bulk touched {r}"
            for r, idx in sel.items():
                assert all(p["cls"] != "scratch" or p["conf"] < 0.85 for p in tab.data["images"][r]["proposals"])
            tab.undo()
            assert sum(len(v) for v in AA.select_proposals(tab.data, tab.view, cls="scratch", lo=0.85).values()) == n_sel
            print(f"ok  bulk accept changed exactly the {n_sel} selected proposals; undo restores them in one step")

            # ---- keyboard: works on the page, never while typing in a text field
            class _Ev:
                def __init__(self, keysym, widget, state=0):
                    self.keysym, self.widget, self.state = keysym, widget, state
            rel2 = tab.rel
            tab._active = lambda: True
            assert tab._on_key(_Ev("n", tab.lo_entry._entry)) is None and tab.rel == rel2, "shortcut fired in a field"
            assert tab._on_key(_Ev("n", tab.canvas)) == "break"
            print("ok  shortcuts act on the canvas, never while typing in an entry")

            # ---- project switch during generation: proposals go to the job's own project file
            first_ann = tab.ann_path
            more = [f"_inbox/s{i:02d}.jpg" for i in range(20)]
            for r in more:
                _cv2.imencode(".jpg", blank)[1].tofile(str(D.IMAGE_ROOT / r))
            tab.refresh()
            gate = threading.Event()

            def gated(img):
                gate.wait(5)
                return [("dent", 0.66, 2, 2, 30, 50)]
            tab.start_job(more, Path(td) / "fake.pt", 0.25,
                          detector_factory=lambda w_, c_: (gated, ["dent"], "gated@1"))
            other = D.use_project(D.create_project("Selftest Other", task="detection"))
            A.save_annotations(D.PROJECT_DIR / "annotations.json", A.init_annotations("detection", ["dent"]))
            (D.IMAGE_ROOT / D.INBOX_DIR).mkdir(parents=True, exist_ok=True)
            _cv2.imencode(".jpg", blank)[1].tofile(str(D.IMAGE_ROOT / D.INBOX_DIR / "o1.jpg"))
            tab.refresh()                                                         # page now shows the other project
            assert tab.ann_path != first_ann
            gate.set()
            wait_job(tab, root)
            mine = A.load_annotations(first_ann)["images"]
            theirs = A.load_annotations(tab.ann_path)["images"]
            assert all((mine.get(r) or {}).get("proposals") for r in more), "proposals lost after project switch"
            assert not any(r in theirs for r in more), "proposals attached to the wrong project"
            print(f"ok  project switched during generation: proposals saved to the job's project, none in {other!r}")
            root.destroy()

            # ---- a classification project WITH annotations.json opens it (labels.csv untouched)
            cls_name = D.use_project(D.create_project("Selftest Cls With Boxes"))
            (D.IMAGE_ROOT / D.INBOX_DIR).mkdir(parents=True, exist_ok=True)
            _cv2.imencode(".jpg", blank)[1].tofile(str(D.IMAGE_ROOT / D.INBOX_DIR / "k1.jpg"))
            d5 = A.init_annotations("detection", ["bottle"])
            AA.propose(d5, "_inbox/k1.jpg", [("bottle", 0.9, 2, 2, 30, 50)], (40, 60))
            A.save_annotations(D.PROJECT_DIR / "annotations.json", d5)
            labels_before = D.LABELS_CSV.read_bytes() if D.LABELS_CSV.exists() else None
            root4 = ctk.CTk()
            root4.withdraw()
            tab4 = AnnotationTab(app=None, parent=ctk.CTkFrame(root4))
            tab4.refresh()
            assert tab4.body.winfo_manager() == "pack" and tab4.rel == "_inbox/k1.jpg"
            tab4.accept_proposals()
            assert (D.LABELS_CSV.read_bytes() if D.LABELS_CSV.exists() else None) == labels_before
            root4.destroy()
            print(f"ok  classification project {cls_name!r} with annotations.json opens for review; labels.csv untouched")

            # ---- segmentation project ---------------------------------------
            D.use_project(D.create_project("Selftest Segmentation", task="segmentation"))
            img2 = D.IMAGE_ROOT / D.INBOX_DIR / "img002.jpg"
            img2.parent.mkdir(parents=True, exist_ok=True)
            _cv2.imencode(".jpg", blank)[1].tofile(str(img2))
            A.save_annotations(D.PROJECT_DIR / "annotations.json", A.init_annotations("segmentation", ["dent"]))
            root2 = ctk.CTk()
            root2.withdraw()
            tab2 = AnnotationTab(app=None, parent=ctk.CTkFrame(root2))
            tab2.refresh()
            assert tab2.task == "segmentation"
            assert str(tab2.auto_btn.cget("state")) == "disabled"
            tab2.poly_points = [0.1, 0.1, 0.2, 0.2]
            tab2.finish_polygon()
            assert tab2._entry()["polygons"] == [], "a 2-point polygon must not be accepted"
            tab2.poly_points = [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]
            tab2.class_menu.set("dent")
            tab2.finish_polygon()
            assert len(tab2._entry()["polygons"]) == 1
            tab2.save()
            print("ok  segmentation: reject <3-point polygon, accept >=3-point polygon, save")
            tab2.selected = ("poly", 0)
            tab2.delete_selected()
            assert tab2._entry()["polygons"] == []
            print("ok  segmentation: select and delete polygon")
            tab2._entry()["polygons"].append({"cls": "dent", "points": [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]})
            tab2.save()
            with tempfile.TemporaryDirectory() as outdir:
                written = A.export_yolo_segmentation(tab2.data, Path(outdir))
                assert written == ["_inbox/img002.jpg"]
            print("ok  segmentation: YOLO export via annotate.py")
            root2.destroy()

            # ---- classification project without annotations.json: notice, no editor
            was_task_default = D.use_project(D.create_project("Selftest Classification"))
            assert D.project_task(was_task_default) == "classification"
            root3 = ctk.CTk()
            root3.withdraw()
            tab3 = AnnotationTab(app=None, parent=ctk.CTkFrame(root3))
            tab3.refresh()
            assert not tab3.body.winfo_manager() == "pack", "classification project must not show the editor"
            assert "not required" in tab3.notice.cget("text")
            root3.destroy()
            print("ok  classification project without annotations shows the not-required notice, no editor")
    finally:
        mb.showerror, mb.showinfo, mb.askyesno = orig
        D.PROJECTS, D.ACTIVE_TXT = was[0], was[1]
        if was[2]:
            D.use_project(was[2])

    print("ok  annotation_studio.py self-test complete -- "
          "no file under the real projects/ directory (including om_bottle) was touched")


if __name__ == "__main__":
    selftest()
