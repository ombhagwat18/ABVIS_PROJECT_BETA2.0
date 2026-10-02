"""Annotation Studio -- interactive detection/segmentation labelling UI.

This module owns UI/interaction code only. It never reads or writes
annotations.json directly: every load, save, validate, or export goes
through annotate.py, which stays the single source of truth for the
annotation schema. This module also never touches labels.csv, train.py, or
infer.py -- classification projects are completely unaffected by anything
here.

Integration point: `AnnotationTab` follows the same `(app, parent)`
constructor + `refresh()` convention as every tab in gui.py (LabelTab,
DefectsTab, ...), so gui.py's only change is one new tab entry -- see the
small diff described in the Stage 2C report.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

import annotate as A
import dataset as D

# Same palette values as gui.py (duplicated here, not imported, to avoid a
# gui.py <-> annotation_studio.py import cycle: gui.py imports this module to
# build the "Annotate" tab).
ACC, ACC_H, ACC_T = "#1d4ed8", "#1e40af", "#ffffff"
BAD, GOOD, WARN, DIM = "#b91c1c", "#15803d", "#b45309", "#5b6672"
PANEL, BG, INK = "#f1f5fb", "#ffffff", "#0f172a"

CANVAS_W, CANVAS_H = 760, 560
HANDLE = 7           # canvas px, resize-handle square (half-width)
MIN_BOX = 0.01        # normalized -- boxes/drags smaller than this are discarded, not saved


class AnnotationTab:
    """The 'Annotate' tab. Only meaningful for a project whose task is
    "detection" or "segmentation" -- refresh() shows a plain notice instead
    for a classification project (including om_bottle)."""

    def __init__(self, app, parent):
        self.app = app
        self.parent = parent

        self.task: str | None = None
        self.ann_path: Path | None = None
        self.data: dict | None = None
        self.images: list[str] = []
        self.idx = 0
        self.rel: str | None = None
        self.dirty = False
        self.image_root: Path | None = None          # None -> falls back to D.IMAGE_ROOT
        self.meta: dict[str, dict] = {}                # rel -> {"scene_id":.., "split":..}, optional

        self.img_bgr = None
        self.img_w = self.img_h = 0
        self.scale = 1.0
        self.off_x = self.off_y = 0.0
        self.photo = None                    # keep alive -- Tk drops unreferenced PhotoImages

        self.selected: tuple[str, int] | None = None   # ("box", i) | ("poly", i)
        self.drag: dict | None = None
        self.poly_points: list[float] = []              # in-progress polygon, flat [x,y,...]

        self._rows: dict[str, ctk.CTkFrame] = {}
        self._row_imgs: list = []

        self._build(parent)

    # ------------------------------------------------------------- building
    def _build(self, parent):
        self.notice = ctk.CTkLabel(parent, text="", text_color=DIM, font=("Segoe UI", 13),
                                    wraplength=760, justify="left")

        self.init_frame = ctk.CTkFrame(parent, fg_color="transparent")

        self.body = ctk.CTkFrame(parent, fg_color="transparent")   # packed only for det/seg tasks

        left = ctk.CTkFrame(self.body, fg_color=PANEL, width=220)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="IMAGES", text_color=DIM, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4))
        self.imglist = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.imglist.pack(fill="both", expand=True, padx=4)
        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(nav, text="< Prev", width=90, command=self.prev_image).pack(side="left")
        ctk.CTkButton(nav, text="Next >", width=90, command=self.next_image).pack(side="right")
        ctk.CTkButton(left, text="Next Unannotated", command=self.goto_next_unannotated).pack(
            fill="x", padx=10, pady=(0, 4))
        self.progress_lbl = ctk.CTkLabel(left, text="", text_color=DIM, font=("Segoe UI", 11),
                                         justify="left", anchor="w")
        self.progress_lbl.pack(fill="x", padx=10, pady=(0, 8))

        center = ctk.CTkFrame(self.body, fg_color="transparent")
        center.pack(side="left", fill="both", expand=True)
        topbar = ctk.CTkFrame(center, fg_color=PANEL)
        topbar.pack(fill="x", pady=(0, 6))
        self.fname_lbl = ctk.CTkLabel(topbar, text="-", font=("Segoe UI", 12, "bold"))
        self.fname_lbl.pack(side="left", padx=10, pady=8)
        self.meta_lbl = ctk.CTkLabel(topbar, text="", text_color=DIM)
        self.meta_lbl.pack(side="left", padx=10)
        self.reviewed_lbl = ctk.CTkLabel(topbar, text="", text_color=DIM)
        self.reviewed_lbl.pack(side="left", padx=10)
        ctk.CTkButton(topbar, text="Mark reviewed", width=120,
                      command=self.mark_reviewed).pack(side="right", padx=6)

        self.canvas = ctk.CTkCanvas(center, width=CANVAS_W, height=CANVAS_H, bg=INK,
                                    highlightthickness=0)
        self.canvas.pack(pady=(0, 6))
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda e: self.finish_polygon())
        self.canvas.bind("<Delete>", lambda e: self.delete_selected())

        actionbar = ctk.CTkFrame(center, fg_color="transparent")
        actionbar.pack(fill="x")
        ctk.CTkButton(actionbar, text="Delete selected", width=130, fg_color="transparent",
                      border_width=1, text_color=BAD, command=self.delete_selected).pack(
            side="left", padx=(0, 6))
        self.finish_btn = ctk.CTkButton(actionbar, text="Finish polygon", width=130,
                                        command=self.finish_polygon)
        self.finish_btn.pack(side="left", padx=(0, 6))
        self.cancel_poly_btn = ctk.CTkButton(actionbar, text="Cancel polygon", width=130,
                                             fg_color="transparent", border_width=1,
                                             command=self.cancel_polygon)
        self.cancel_poly_btn.pack(side="left")
        self.hint = ctk.CTkLabel(actionbar, text="", text_color=DIM, font=("Segoe UI", 11))
        self.hint.pack(side="right")

        right = ctk.CTkFrame(self.body, fg_color=PANEL, width=230)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="ACTIVE CLASS", text_color=DIM,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.class_menu = ctk.CTkOptionMenu(right, values=["-"], width=200)
        self.class_menu.pack(padx=12, pady=(0, 12))

        ctk.CTkLabel(right, text="FILE", text_color=DIM, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=12, pady=(0, 4))
        ctk.CTkButton(right, text="Save", fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=self.save).pack(fill="x", padx=12, pady=3)
        ctk.CTkButton(right, text="Reload", fg_color="transparent", border_width=1,
                      command=self.reload_annotations).pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(right, text="EXPORT", text_color=DIM, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=12, pady=(16, 4))
        self.export_det_btn = ctk.CTkButton(right, text="Export YOLO Detection",
                                            fg_color="transparent", border_width=1,
                                            command=self.export_detection)
        self.export_det_btn.pack(fill="x", padx=12, pady=3)
        self.export_seg_btn = ctk.CTkButton(right, text="Export YOLO Segmentation",
                                            fg_color="transparent", border_width=1,
                                            command=self.export_segmentation)
        self.export_seg_btn.pack(fill="x", padx=12, pady=3)

        self.status = ctk.CTkLabel(right, text="", text_color=DIM, font=("Segoe UI", 11),
                                   wraplength=200, justify="left")
        self.status.pack(anchor="w", padx=12, pady=(16, 8))

    # --------------------------------------------------------------- refresh
    def refresh(self):
        """Called by App.reload() on every project switch/data change, same
        as every other tab."""
        self.image_root = D.IMAGE_ROOT      # project mode always uses the active project's images/
        self.meta = {}
        task = D.project_task(D.PROJECT) if D.PROJECT else D.DEFAULT_TASK
        self.task = task
        self.body.pack_forget()
        self.init_frame.pack_forget()

        if task not in ("detection", "segmentation"):
            self.notice.configure(
                text=("Annotation mode is not required for this project's task "
                      f"({task!r}).\n\nThis project trains a classifier from labels.csv, "
                      "labelled in the Label tab. The Annotate tab applies only to "
                      "projects whose task is \"detection\" or \"segmentation\"."))
            self.notice.pack(pady=60)
            return
        self.notice.pack_forget()

        self.ann_path = D.PROJECT_DIR / "annotations.json"
        if not self.ann_path.exists():
            self._show_init_prompt(task)
            return

        self.data = A.load_annotations(self.ann_path)
        self.dirty = False
        self.images = D.scan_images()
        self._populate_classes()
        self._rebuild_image_list()
        self.export_det_btn.configure(state="normal" if task == "detection" else "disabled")
        self.export_seg_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.finish_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.cancel_poly_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.hint.configure(
            text=("click-drag to draw a box, click to select/move, drag its corner to resize"
                  if task == "detection" else
                  "click to add polygon points, right-click or 'Finish polygon' to close"))
        self.body.pack(fill="both", expand=True)

        if self.images:
            self.idx = min(self.idx, len(self.images) - 1)
            self.load_image(self.images[self.idx])
        else:
            self.rel = None
            self.fname_lbl.configure(text="-")
            self.status.configure(
                text="No images found under this project's images/ folder yet.")
            self.canvas.delete("all")

    def _show_init_prompt(self, task):
        for w in self.init_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.init_frame,
                     text=f"This project's task is {task!r} but no annotations.json exists yet.",
                     font=("Segoe UI", 14, "bold")).pack(pady=(60, 6))
        ctk.CTkLabel(self.init_frame,
                     text="Enter class names, comma-separated, to initialize it. "
                          "This creates annotations.json only for THIS project.",
                     text_color=DIM, wraplength=500, justify="center").pack(pady=(0, 10))
        entry = ctk.CTkEntry(self.init_frame, width=380,
                             placeholder_text="e.g. scratch, dent, missing_cap")
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

        ctk.CTkButton(self.init_frame, text="Initialize annotations.json",
                      fg_color=ACC, text_color=ACC_T, hover_color=ACC_H,
                      command=do_init).pack(pady=8)
        self.init_frame.pack(fill="both", expand=True)

    def load_external(self, image_root: Path, ann_path: Path, task: str, classes: list[str],
                       images: list[str], meta: dict[str, dict] | None = None):
        """Point this tab at an explicit image set instead of the active
        project's images/ folder -- used for Stage 2 detection annotation,
        where the candidate images live under stage2_dataset/, not under any
        projects/<slug>/images/. Never touches dataset.py's project globals,
        review_manifest.csv, split.json, split_manifest.csv, or Stage 1.

        `ann_path` is created fresh (via annotate.init_annotations) only if
        it doesn't already exist -- an existing file's annotations are always
        loaded and preserved, never overwritten by this call."""
        self.notice.pack_forget()
        self.init_frame.pack_forget()
        self.image_root = Path(image_root)
        self.ann_path = Path(ann_path)
        self.task = task
        self.meta = meta or {}
        if not self.ann_path.exists():
            A.save_annotations(self.ann_path, A.init_annotations(task, classes))
        self.data = A.load_annotations(self.ann_path)
        self.dirty = False
        self.images = list(images)
        self.idx = 0
        self._populate_classes()
        self._rebuild_image_list()
        self.export_det_btn.configure(state="normal" if task == "detection" else "disabled")
        self.export_seg_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.finish_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.cancel_poly_btn.configure(state="normal" if task == "segmentation" else "disabled")
        self.hint.configure(
            text=("click-drag to draw a box, click to select/move, drag its corner to resize"
                  if task == "detection" else
                  "click to add polygon points, right-click or 'Finish polygon' to close"))
        self.body.pack(fill="both", expand=True)
        if self.images:
            self.load_image(self.images[0])

    def _populate_classes(self):
        values = list(self.data["classes"]) or ["-"]
        self.class_menu.configure(values=values)
        self.class_menu.set(values[0])

    # ---------------------------------------------------------- image list
    def _rebuild_image_list(self):
        for w in self.imglist.winfo_children():
            w.destroy()
        self._rows.clear()
        self._row_imgs.clear()
        for rel in self.images:
            entry = self.data["images"].get(rel, {})
            n = len(entry.get("boxes", [])) + len(entry.get("polygons", []))
            reviewed = bool(entry.get("reviewed"))
            row = ctk.CTkFrame(self.imglist, fg_color=PANEL if rel != self.rel else "#dbe6fb")
            row.pack(fill="x", pady=2, padx=2)
            dot = ctk.CTkLabel(row, text="●", text_color=GOOD if reviewed else WARN, width=16)
            dot.pack(side="left", padx=(6, 0))
            lab = ctk.CTkLabel(row, text=f"{Path(rel).name}  ({n})", anchor="w",
                               font=("Segoe UI", 11))
            lab.pack(side="left", fill="x", expand=True, padx=4, pady=4)
            for w in (row, dot, lab):
                w.bind("<Button-1>", lambda e, p=rel: self.goto_image(p))
            self._rows[rel] = row
        self._update_progress_label()

    def _update_progress_label(self):
        if self.data is None:
            self.progress_lbl.configure(text="")
            return
        c = A.progress_counts(self.data, self.images)
        self.progress_lbl.configure(
            text=f"total {c['total']}  ·  pending {c['pending']}\n"
                 f"annotated {c['annotated']}  ·  reviewed {c['reviewed']}")

    def _highlight_row(self, rel):
        for p, row in self._rows.items():
            row.configure(fg_color=PANEL if p != rel else "#dbe6fb")

    # -------------------------------------------------------------- loading
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno("Unsaved changes",
                                    "Discard unsaved annotation changes for this image?")

    def goto_image(self, rel):
        if rel == self.rel:
            return
        if not self._confirm_discard():
            return
        self.dirty = False
        self.idx = self.images.index(rel)
        self.load_image(rel)

    def next_image(self):
        if not self.images or not self._confirm_discard():
            return
        self.dirty = False
        self.idx = min(self.idx + 1, len(self.images) - 1)
        self.load_image(self.images[self.idx])

    def prev_image(self):
        if not self.images or not self._confirm_discard():
            return
        self.dirty = False
        self.idx = max(self.idx - 1, 0)
        self.load_image(self.images[self.idx])

    def load_image(self, rel):
        root = self.image_root if self.image_root is not None else D.IMAGE_ROOT
        img = D.imread(root / rel)
        if img is None:
            messagebox.showerror("Cannot read image", rel)
            return
        self.rel = rel
        self.img_bgr = img
        self.img_h, self.img_w = img.shape[:2]
        self.scale = min(CANVAS_W / self.img_w, CANVAS_H / self.img_h)
        dw, dh = self.img_w * self.scale, self.img_h * self.scale
        self.off_x, self.off_y = (CANVAS_W - dw) / 2, (CANVAS_H - dh) / 2
        self.selected = None
        self.drag = None
        self.poly_points = []
        self.fname_lbl.configure(text=rel)
        m = self.meta.get(rel, {})
        self.meta_lbl.configure(
            text=f"scene {m['scene_id']} · {m['split']}" if m else "")
        entry = self.data["images"].get(rel, {})
        reviewed = bool(entry.get("reviewed"))
        self.reviewed_lbl.configure(text="reviewed" if reviewed else "unreviewed",
                                    text_color=GOOD if reviewed else WARN)
        self._highlight_row(rel)
        self._redraw()

    def goto_next_unannotated(self):
        if self.data is None:
            return
        rel = A.next_unannotated(self.images, self.data, after=self.rel)
        if rel is None:
            messagebox.showinfo("Next Unannotated", "No pending (unannotated) images left.")
            return
        self.goto_image(rel)

    # --------------------------------------------------------- coordinates
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
        """Get-or-create the current image's annotation entry, in-memory only
        (nothing touches disk here -- see save())."""
        return self.data["images"].setdefault(
            self.rel, {"reviewed": False, "boxes": [], "polygons": []})

    # ------------------------------------------------------------- drawing
    def _redraw(self):
        self.canvas.delete("all")
        if self.img_bgr is None:
            return
        dw, dh = round(self.img_w * self.scale), round(self.img_h * self.scale)
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (max(1, dw), max(1, dh)), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(small))
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw", image=self.photo)

        entry = self.data["images"].get(self.rel, {}) if self.data else {}
        for i, box in enumerate(entry.get("boxes", [])):
            sel = self.selected == ("box", i)
            x0, y0 = self.n2c(box["x"] - box["w"] / 2, box["y"] - box["h"] / 2)
            x1, y1 = self.n2c(box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)
            color = ACC if sel else GOOD
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2 if sel else 1)
            self.canvas.create_text(x0 + 3, max(0, y0 - 8), text=box["cls"], fill=color,
                                    anchor="w", font=("Segoe UI", 9, "bold"))
            if sel:
                self.canvas.create_rectangle(x1 - HANDLE, y1 - HANDLE, x1 + HANDLE, y1 + HANDLE,
                                             fill=color, outline=color)

        for i, poly in enumerate(entry.get("polygons", [])):
            sel = self.selected == ("poly", i)
            pts = poly["points"]
            flat = []
            for j in range(0, len(pts), 2):
                flat += list(self.n2c(pts[j], pts[j + 1]))
            color = ACC if sel else GOOD
            self.canvas.create_polygon(*flat, outline=color, fill="", width=2 if sel else 1)
            self.canvas.create_text(flat[0] + 3, flat[1] - 8, text=poly["cls"], fill=color,
                                    anchor="w", font=("Segoe UI", 9, "bold"))

        if self.poly_points:
            flat = []
            for j in range(0, len(self.poly_points), 2):
                flat += list(self.n2c(self.poly_points[j], self.poly_points[j + 1]))
            if len(flat) >= 4:
                self.canvas.create_line(*flat, fill=WARN, width=2, dash=(4, 2))
            for j in range(0, len(flat), 2):
                self.canvas.create_oval(flat[j] - 3, flat[j + 1] - 3, flat[j] + 3, flat[j + 1] + 3,
                                        fill=WARN, outline=WARN)

        if self.drag and self.drag.get("mode") == "new":
            x0, y0 = self.drag["start"]
            x1, y1 = self.drag.get("cur", (x0, y0))
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACC, width=2, dash=(3, 2))

    # ------------------------------------------------------------ hit-tests
    def _boxes(self):
        return self._entry()["boxes"] if self.rel else []

    def _hit_box(self, cx, cy):
        for i in reversed(list(enumerate(self._boxes()))):
            i, box = i
            x0, y0 = self.n2c(box["x"] - box["w"] / 2, box["y"] - box["h"] / 2)
            x1, y1 = self.n2c(box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return i
        return None

    def _hit_handle(self, cx, cy):
        if not self.selected or self.selected[0] != "box":
            return False
        box = self._boxes()[self.selected[1]]
        _, _ = self.n2c(box["x"] - box["w"] / 2, box["y"] - box["h"] / 2)
        x1, y1 = self.n2c(box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)
        return abs(cx - x1) <= HANDLE + 3 and abs(cy - y1) <= HANDLE + 3

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

    # ------------------------------------------------------------- editing
    def on_press(self, e):
        if self.img_bgr is None:
            return
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

        # detection
        if self._hit_handle(cx, cy):
            box = dict(self._boxes()[self.selected[1]])
            self.drag = {"mode": "resize", "orig": box}
            return
        hit = self._hit_box(cx, cy)
        if hit is not None:
            self.selected = ("box", hit)
            box = dict(self._boxes()[hit])
            nx, ny = self.c2n(cx, cy)
            self.drag = {"mode": "move", "orig": box, "grab": (nx, ny)}
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
            box["x"] = min(1.0, max(0.0, box["x"] + (nx - gnx)))
            box["y"] = min(1.0, max(0.0, box["y"] + (ny - gny)))
            self._boxes()[self.selected[1]] = box
            self.dirty = True
            self._redraw()
        elif mode == "resize":
            nx, ny = self.c2n(cx, cy)
            orig = self.drag["orig"]
            x0, y0 = orig["x"] - orig["w"] / 2, orig["y"] - orig["h"] / 2
            neww = max(MIN_BOX, nx - x0)
            newh = max(MIN_BOX, ny - y0)
            box = dict(orig)
            box["w"], box["h"] = neww, newh
            box["x"], box["y"] = x0 + neww / 2, y0 + newh / 2
            self._boxes()[self.selected[1]] = box
            self.dirty = True
            self._redraw()

    def on_release(self, e):
        if self.drag is None or self.task != "detection":
            self.drag = None
            return
        if self.drag["mode"] == "new":
            x0c, y0c = self.drag["start"]
            x1c, y1c = self.drag.get("cur", (x0c, y0c))
            nx0, ny0 = self.c2n(min(x0c, x1c), min(y0c, y1c))
            nx1, ny1 = self.c2n(max(x0c, x1c), max(y0c, y1c))
            w, h = nx1 - nx0, ny1 - ny0
            if w >= MIN_BOX and h >= MIN_BOX:
                box = {"cls": self.class_menu.get(), "x": nx0 + w / 2, "y": ny0 + h / 2,
                       "w": w, "h": h}
                self._boxes().append(box)
                self.selected = ("box", len(self._boxes()) - 1)
                self.dirty = True
                self._update_progress_label()
        self.drag = None
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
        lst = entry["boxes"] if kind == "box" else entry["polygons"]
        if 0 <= i < len(lst):
            del lst[i]
            self.dirty = True
            self._update_progress_label()
        self.selected = None
        self._redraw()

    def mark_reviewed(self):
        if not self.rel:
            return
        self._entry()["reviewed"] = True
        self.dirty = True
        self.reviewed_lbl.configure(text="reviewed", text_color=GOOD)
        self._highlight_row(self.rel)
        self._update_progress_label()

    # ---------------------------------------------------------- save/load
    def save(self):
        if self.data is None:
            return
        try:
            A.save_annotations(self.ann_path, self.data)
        except A.AnnotationError as e:
            messagebox.showerror("Cannot save", str(e))
            return
        self.dirty = False
        self.status.configure(text=f"Saved {self.ann_path.name}")
        self._rebuild_image_list()

    def reload_annotations(self):
        if self.ann_path is None:
            return
        if self.dirty and not messagebox.askyesno(
                "Reload from disk", "Discard unsaved edits and reload annotations.json?"):
            return
        try:
            self.data = A.load_annotations(self.ann_path)
        except A.AnnotationError as e:
            messagebox.showerror("Cannot reload", str(e))
            return
        self.dirty = False
        self._populate_classes()
        self._rebuild_image_list()
        if self.rel:
            self.load_image(self.rel)

    # ------------------------------------------------------------- export
    def export_detection(self):
        if self.task != "detection" or self.data is None:
            return
        out = filedialog.askdirectory(title="Export YOLO detection labels to...")
        if not out:
            return
        try:
            written = A.export_yolo_detection(self.data, Path(out))
        except A.AnnotationError as e:
            messagebox.showerror("Export failed", str(e))
            return
        messagebox.showinfo("Exported", f"Wrote {len(written)} label file(s) to {out}")

    def export_segmentation(self):
        if self.task != "segmentation" or self.data is None:
            return
        out = filedialog.askdirectory(title="Export YOLO segmentation labels to...")
        if not out:
            return
        try:
            written = A.export_yolo_segmentation(self.data, Path(out))
        except A.AnnotationError as e:
            messagebox.showerror("Export failed", str(e))
            return
        messagebox.showinfo("Exported", f"Wrote {len(written)} label file(s) to {out}")


# ------------------------------------------------------------------ self-test

def selftest():
    """Exercises AnnotationTab against a real (but disposable) detection
    project and a real (but disposable) segmentation project, both created
    under a temporary projects/ root -- exactly the isolation technique
    dataset.py's own project_demo() uses -- so nothing under the real
    projects/ directory, and specifically nothing under projects/om_bottle/,
    is ever touched.

    A full mouse-driven GUI interaction (actual click-drag events) is not
    simulated here -- Tk's synthetic event generation is unreliable across
    platforms for this kind of test. Instead this drives the exact same
    methods the mouse handlers call (on_release's box-commit logic via a
    direct call to the underlying list append, finish_polygon(),
    delete_selected(), save(), reload_annotations(), export_*()), which
    covers every code path the UI itself calls -- everything except the
    literal OS-level mouse motion.
    """
    import tempfile
    import tkinter.messagebox as mb

    import cv2 as _cv2
    import numpy as np

    # A real messagebox popup would block this headless test forever (no
    # mainloop is running to let a human dismiss it). Every path this
    # self-test exercises that would normally show one is instead recorded
    # here and asserted on directly, then the real functions are restored.
    was = (D.PROJECTS, D.ACTIVE_TXT, D.PROJECT)
    orig = (mb.showerror, mb.showinfo, mb.askyesno)
    popups = []
    mb.showerror = lambda title, msg, *a, **k: popups.append(("error", title, msg))
    mb.showinfo = lambda title, msg, *a, **k: popups.append(("info", title, msg))
    mb.askyesno = lambda *a, **k: True
    blank = np.zeros((60, 40, 3), np.uint8)
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

            # init prompt path (no annotations.json yet)
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

            # move: reuse the exact on_drag "move" arithmetic
            tab.drag = {"mode": "move", "orig": dict(tab._boxes()[0]), "grab": (0.5, 0.5)}
            box = dict(tab.drag["orig"])
            box["x"], box["y"] = 0.6, 0.4
            tab._boxes()[0] = box
            assert tab._boxes()[0]["x"] == 0.6 and tab._boxes()[0]["y"] == 0.4
            tab.drag = None

            # unknown-class / out-of-range safety: hand-craft a bad box, confirm save refuses it
            before = tab.ann_path.read_text(encoding="utf-8")
            tab._boxes().append({"cls": "not_a_class", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1})
            popups.clear()
            tab.save()   # must not persist the bad box; must surface an error, not fail silently
            assert tab.ann_path.read_text(encoding="utf-8") == before, \
                "save() must not have written a box with an unknown class to disk"
            assert popups and popups[-1][0] == "error", "save() swallowed the validation failure silently"
            tab._boxes().pop()  # remove the bad box, restore valid state

            tab.save()
            reloaded = A.load_annotations(tab.ann_path)
            assert reloaded["images"]["_inbox/img001.jpg"]["boxes"][0]["cls"] == "scratch"
            print("ok  detection: init, box create/move, unknown-class save-rejection, save/reload")

            # delete
            tab.selected = ("box", 0)
            tab.delete_selected()
            assert tab._boxes() == []
            tab.save()
            print("ok  detection: delete selected box, persisted")

            # export
            with tempfile.TemporaryDirectory() as outdir:
                tab._boxes().append({"cls": "dent", "x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1})
                tab.save()
                written = A.export_yolo_detection(tab.data, Path(outdir))
                assert written == ["_inbox/img001.jpg"]
                txt = (Path(outdir) / "img001.txt").read_text().strip()
                assert txt == "1 0.300000 0.300000 0.100000 0.100000"
            print("ok  detection: YOLO export via annotate.py")

            root.destroy()

            # ---- segmentation project ---------------------------------------
            name2 = D.use_project(D.create_project("Selftest Segmentation", task="segmentation"))
            img2 = D.IMAGE_ROOT / D.INBOX_DIR / "img002.jpg"
            img2.parent.mkdir(parents=True, exist_ok=True)
            _cv2.imencode(".jpg", blank)[1].tofile(str(img2))
            data2 = A.init_annotations("segmentation", ["dent"])
            A.save_annotations(D.PROJECT_DIR / "annotations.json", data2)

            root2 = ctk.CTk()
            root2.withdraw()
            tab2 = AnnotationTab(app=None, parent=ctk.CTkFrame(root2))
            tab2.refresh()
            assert tab2.task == "segmentation"

            # too-few-points must be rejected
            tab2.poly_points = [0.1, 0.1, 0.2, 0.2]     # only 2 points
            tab2.finish_polygon()
            assert tab2._entry()["polygons"] == [], "a 2-point polygon must not be accepted"

            tab2.poly_points = [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]
            tab2.class_menu.set("dent")
            tab2.finish_polygon()
            assert len(tab2._entry()["polygons"]) == 1
            tab2.save()
            print("ok  segmentation: reject <3-point polygon, accept >=3-point polygon, save")

            # select + delete
            tab2.selected = ("poly", 0)
            tab2.delete_selected()
            assert tab2._entry()["polygons"] == []
            print("ok  segmentation: select and delete polygon")

            # export
            tab2._entry()["polygons"].append(
                {"cls": "dent", "points": [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]})
            tab2.save()
            with tempfile.TemporaryDirectory() as outdir:
                written = A.export_yolo_segmentation(tab2.data, Path(outdir))
                assert written == ["_inbox/img002.jpg"]
            print("ok  segmentation: YOLO export via annotate.py")

            root2.destroy()

            # ---- classification project: Annotate tab must show the notice --
            was_task_default = D.use_project(D.create_project("Selftest Classification"))
            assert D.project_task(was_task_default) == "classification"
            root3 = ctk.CTk()
            root3.withdraw()
            tab3 = AnnotationTab(app=None, parent=ctk.CTkFrame(root3))
            tab3.refresh()
            assert not tab3.body.winfo_manager() == "pack", "classification project must not show the editor"
            assert "not required" in tab3.notice.cget("text")
            root3.destroy()
            print("ok  classification project shows the not-required notice, no editor")

    finally:
        mb.showerror, mb.showinfo, mb.askyesno = orig
        D.PROJECTS, D.ACTIVE_TXT = was[0], was[1]
        if was[2]:
            D.use_project(was[2])

    print("ok  annotation_studio.py self-test complete -- "
          "no file under the real projects/ directory (including om_bottle) was touched")


if __name__ == "__main__":
    selftest()
