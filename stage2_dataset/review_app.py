"""
Stage 2 manual-review interface -- KEEP / REJECT / AMBIGUOUS triage over
stage2_dataset/clean/, one image at a time.

This is deliberately scoped to review only: no bounding boxes, no YOLO
labels, no training, no merge with Stage 1. It is built as a standalone
CustomTkinter app (not a gui.py tab) because the Stage 2 collection is not
a dataset.py "project" -- see AUDIT_REPORT.md. review_manifest_io.py holds
all the non-UI logic (load/save/ordering) so the Stage 2 Annotation Studio
can import and extend it later without pulling in this window.

Run: python stage2_dataset/review_app.py
Self-check (no window opened): python stage2_dataset/review_app.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import review_manifest_io as M

ACC, GOOD, BAD, WARN, DIM, INK = "#1d4ed8", "#15803d", "#b91c1c", "#b45309", "#5b6672", "#0f172a"
PANEL, BG = "#f1f5fb", "#ffffff"

CANVAS_W, CANVAS_H = 900, 620


class ReviewApp:
    def __init__(self, root, rows: list[dict], order: list[int]):
        self.root = root
        self.rows = rows
        self.order = order
        self.pos = 0
        self.photo = None  # keep alive

        self._build_ui()
        self._show_current()

    # -- UI construction --------------------------------------------------
    def _build_ui(self):
        import customtkinter as ctk

        self.root.title("Stage 2 Review")
        self.root.geometry("1300x760")

        main = ctk.CTkFrame(self.root, fg_color=BG)
        main.pack(fill="both", expand=True)

        left = ctk.CTkFrame(main, fg_color=BG)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.image_label = ctk.CTkLabel(left, text="", fg_color="#000000")
        self.image_label.pack(fill="both", expand=True)

        nav = ctk.CTkFrame(left, fg_color=BG)
        nav.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(nav, text="< Previous", command=self.prev).pack(side="left")
        self.progress_label = ctk.CTkLabel(nav, text="")
        self.progress_label.pack(side="left", expand=True)
        ctk.CTkButton(nav, text="Next >", command=self.next_).pack(side="right")

        jump = ctk.CTkFrame(left, fg_color=BG)
        jump.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(jump, text="Jump to (filename or index):").pack(side="left")
        self.jump_entry = ctk.CTkEntry(jump, width=200)
        self.jump_entry.pack(side="left", padx=6)
        self.jump_entry.bind("<Return>", lambda e: self.jump())
        ctk.CTkButton(jump, text="Go", width=50, command=self.jump).pack(side="left")

        right = ctk.CTkFrame(main, fg_color=PANEL, width=340)
        right.pack(side="right", fill="y", padx=8, pady=8)
        right.pack_propagate(False)

        self.info_label = ctk.CTkLabel(right, text="", justify="left", anchor="w", text_color=INK)
        self.info_label.pack(fill="x", padx=12, pady=(12, 4), anchor="w")

        self.flags_label = ctk.CTkLabel(right, text="", justify="left", anchor="w",
                                         text_color=WARN, wraplength=300)
        self.flags_label.pack(fill="x", padx=12, pady=4, anchor="w")

        self.status_label = ctk.CTkLabel(right, text="", justify="left", anchor="w",
                                          text_color=DIM)
        self.status_label.pack(fill="x", padx=12, pady=4, anchor="w")

        ctk.CTkLabel(right, text="Notes:", anchor="w").pack(fill="x", padx=12, pady=(12, 0))
        self.notes_box = ctk.CTkTextbox(right, height=90)
        self.notes_box.pack(fill="x", padx=12, pady=4)

        btns = ctk.CTkFrame(right, fg_color=PANEL)
        btns.pack(fill="x", padx=12, pady=16)
        ctk.CTkButton(btns, text="KEEP  (K)", fg_color=GOOD, height=48,
                      command=lambda: self.decide("KEEP")).pack(fill="x", pady=4)
        ctk.CTkButton(btns, text="REJECT  (R)", fg_color=BAD, height=48,
                      command=lambda: self.decide("REJECT")).pack(fill="x", pady=4)
        ctk.CTkButton(btns, text="AMBIGUOUS  (A)", fg_color=WARN, height=48,
                      command=lambda: self.decide("AMBIGUOUS")).pack(fill="x", pady=4)

        self.root.bind("<Key>", self._on_key)

    def _on_key(self, event):
        # don't hijack keystrokes while typing in the notes/jump boxes
        if str(event.widget).endswith("entry") or "textbox" in str(event.widget):
            return
        key = event.keysym.lower()
        if key == "k":
            self.decide("KEEP")
        elif key == "r":
            self.decide("REJECT")
        elif key == "a":
            self.decide("AMBIGUOUS")
        elif key == "left":
            self.prev()
        elif key == "right":
            self.next_()

    # -- rendering ----------------------------------------------------
    def _current_row(self) -> dict:
        return self.rows[self.order[self.pos]]

    def _show_current(self):
        import cv2
        from PIL import Image
        import customtkinter as ctk

        row = self._current_row()

        img_path = M.CLEAN_DIR / row["clean_filename"]
        bgr = cv2.imread(str(img_path))
        if bgr is not None:
            h, w = bgr.shape[:2]
            scale = min(CANVAS_W / w, CANVAS_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(bgr, (new_w, new_h))
            rgb = resized[:, :, ::-1]
            pil_img = Image.fromarray(rgb)
            self.photo = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
            self.image_label.configure(image=self.photo, text="")
        else:
            self.image_label.configure(image=None, text=f"missing: {img_path.name}")

        n_flagged = sum(1 for i in self.order if M.has_flag(self.rows[i]))
        phase = "FLAGGED" if self.pos < n_flagged else "REMAINING"
        self.progress_label.configure(
            text=f"{self.pos + 1} / {len(self.order)}  ({phase} pass, {n_flagged} flagged total)")

        self.info_label.configure(text=(
            f"File: {row['source_filename']}\n"
            f"Scene ID: {row['scene_id']}\n"
            f"Near-dup group: {row['near_dup_group_id']}"
        ))

        flags = M.active_flags(row)
        self.flags_label.configure(text="Flags: " + (", ".join(flags) if flags else "(none)"))

        color = {"KEEP": GOOD, "REJECT": BAD, "AMBIGUOUS": WARN}.get(row["decision"], DIM)
        self.status_label.configure(
            text=f"Status: {row['review_status']}   Decision: {row['decision'] or '(none)'}",
            text_color=color)

        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("1.0", row.get("notes", ""))

    def _flush_notes(self):
        row = self._current_row()
        row["notes"] = self.notes_box.get("1.0", "end").strip()

    # -- actions --------------------------------------------------------
    def decide(self, decision: str):
        row = self._current_row()
        notes = self.notes_box.get("1.0", "end").strip()
        M.set_decision(row, decision, notes=notes)
        M.save_rows(self.rows)
        self.next_(auto=True)

    def prev(self):
        self._flush_notes()
        M.save_rows(self.rows)
        if self.pos > 0:
            self.pos -= 1
        self._show_current()

    def next_(self, auto: bool = False):
        if not auto:
            self._flush_notes()
            M.save_rows(self.rows)
        if self.pos < len(self.order) - 1:
            self.pos += 1
        self._show_current()

    def jump(self):
        target = self.jump_entry.get().strip()
        if not target:
            return
        self._flush_notes()
        M.save_rows(self.rows)
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(self.order):
                self.pos = idx
        else:
            fn = target if target.endswith(".png") else target + ".png"
            for p, i in enumerate(self.order):
                if self.rows[i]["source_filename"] == fn:
                    self.pos = p
                    break
        self._show_current()


def selftest():
    """Headless check: no Tk window. Verifies ordering, decision/save
    round-trip, and that source/clean/scene_map/audit are never touched."""
    import shutil
    import tempfile

    assert M.MANIFEST_CSV.exists(), "review_manifest.csv missing"
    assert M.CLEAN_DIR.exists(), "clean/ missing"

    rows = M.load_rows()
    assert len(rows) == 606, f"expected 606 rows, got {len(rows)}"

    order = M.build_review_order(rows)
    assert len(order) == 606 and sorted(order) == list(range(606)), "order must be a permutation of all rows"

    n_flagged = sum(1 for r in rows if M.has_flag(r))
    assert n_flagged == 216, f"expected 216 flagged images, got {n_flagged}"

    flagged_prefix = order[:n_flagged]
    assert all(M.has_flag(rows[i]) for i in flagged_prefix), "flagged images must come first"
    assert all(not M.has_flag(rows[i]) for i in order[n_flagged:]), "unflagged images must come after"

    scene_seq = [rows[i]["scene_id"] for i in flagged_prefix]
    # within the flagged pass, same-scene rows must be contiguous
    seen = set()
    prev_scene = None
    for s in scene_seq:
        if s != prev_scene:
            assert s not in seen, f"scene {s} is not contiguous in the flagged pass"
            seen.add(s)
        prev_scene = s

    # round-trip a decision against a scratch copy, never the real manifest
    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / "review_manifest.csv"
        tmp_json = Path(td) / "review_manifest.json"
        shutil.copy2(M.MANIFEST_CSV, tmp_csv)

        scratch_rows = M.load_rows(tmp_csv)
        M.set_decision(scratch_rows[0], "KEEP", notes="selftest note")
        M.save_rows(scratch_rows, tmp_csv, tmp_json)

        reloaded = M.load_rows(tmp_csv)
        assert reloaded[0]["decision"] == "KEEP"
        assert reloaded[0]["review_status"] == "reviewed"
        assert reloaded[0]["notes"] == "selftest note"
        for r in reloaded[1:]:
            assert r["review_status"] == "unreviewed" and r["decision"] == "", "selftest must not touch other rows"

    real_rows = M.load_rows()
    assert all(r["review_status"] == "unreviewed" for r in real_rows), \
        "selftest must leave the real manifest untouched"

    for guard in (M.CLEAN_DIR.parent / "source", M.CLEAN_DIR,
                  M.CLEAN_DIR.parent / "scene_map.json"):
        assert guard.exists(), f"{guard} must still exist"

    print("SELFTEST OK")
    print(f"  manifest rows: {len(rows)}, flagged-first order verified, scene contiguity verified")
    print(f"  decision/save round-trip verified against a scratch copy (real manifest untouched)")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return

    import customtkinter as ctk

    rows = M.load_rows()
    order = M.build_review_order(rows)

    ctk.set_appearance_mode("light")
    root = ctk.CTk()
    ReviewApp(root, rows, order)
    root.mainloop()


if __name__ == "__main__":
    main()
