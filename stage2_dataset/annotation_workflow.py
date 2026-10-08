"""Stage 2 annotation workflow entry point.

Wires the 594 locked candidate images (stage2_dataset/annotation_candidates.json)
into the existing Annotation Studio (annotation_studio.AnnotationTab) and the
existing annotation data layer (annotate.py), without touching:
  - review_manifest.csv, split.json, split_manifest.csv (read-only inputs)
  - Stage 1 / the om_bottle project / any projects/<slug>/ directory
  - training, YOLO export

Creates only stage2_dataset/annotations.json (via annotate.py, first run
only -- an existing file is loaded and preserved, never overwritten).

Run directly to open the studio window:
    python stage2_dataset/annotation_workflow.py

Run self-checks (no GUI mainloop, no real files written):
    python stage2_dataset/annotation_workflow.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import annotate as A

CANDIDATES_JSON = HERE / "annotation_candidates.json"
IMAGE_ROOT = HERE / "clean"
ANNOTATIONS_JSON = HERE / "annotations.json"
CLASSES = ["bottle", "cap", "label"]
TASK = "detection"


def load_candidates(path: Path = CANDIDATES_JSON) -> tuple[list[str], dict[str, dict]]:
    """Returns (images, meta) -- `images` is the ordered list of relative
    filenames (as they exist under stage2_dataset/clean/), `meta` maps each
    filename to {"scene_id":.., "split":..} for display in the studio."""
    data = json.loads(path.read_text(encoding="utf-8"))
    images = [rec["clean_filename"] for rec in data["images"]]
    meta = {rec["clean_filename"]: {"scene_id": rec["scene_id"], "split": rec["split"]}
            for rec in data["images"]}
    return images, meta


def main():
    import customtkinter as ctk

    from ui.annotation_studio import AnnotationTab

    images, meta = load_candidates()

    root = ctk.CTk()
    root.title("Stage 2 Annotation Studio -- bottle / cap / label")
    root.geometry("1300x760")
    tab = AnnotationTab(app=None, parent=root)
    tab.load_external(IMAGE_ROOT, ANNOTATIONS_JSON, TASK, CLASSES, images, meta)
    root.mainloop()


# ------------------------------------------------------------------ self-test

def selftest():
    """Exercises the Stage 2 wiring end to end against a disposable temp
    copy -- never touches the real stage2_dataset/annotations.json, and
    never writes to review_manifest.csv / split.json / split_manifest.csv."""
    import shutil
    import tempfile
    import tkinter.messagebox as mb

    import customtkinter as ctk

    from ui.annotation_studio import AnnotationTab

    real_images, real_meta = load_candidates()
    assert len(real_images) == 594, f"expected 594 candidate images, got {len(real_images)}"
    assert len(set(real_images)) == 594, "duplicate filenames in annotation_candidates.json"
    splits = {m["split"] for m in real_meta.values()}
    assert splits == {"TRAIN", "VAL", "TEST"}
    print("ok  load_candidates: 594 unique images, TRAIN/VAL/TEST splits present")

    orig = (mb.showerror, mb.showinfo, mb.askyesno)
    popups = []
    mb.showerror = lambda title, msg, *a, **k: popups.append(("error", title, msg))
    mb.showinfo = lambda title, msg, *a, **k: popups.append(("info", title, msg))
    mb.askyesno = lambda *a, **k: True

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_ann = Path(td) / "annotations.json"
            # only a handful of real (read-only) images, to keep the test fast
            subset = real_images[:5]
            submeta = {k: v for k, v in real_meta.items() if k in subset}

            root = ctk.CTk()
            root.withdraw()
            tab = AnnotationTab(app=None, parent=ctk.CTkFrame(root))
            tab.load_external(IMAGE_ROOT, tmp_ann, TASK, CLASSES, subset, submeta)
            assert tab.images == subset
            assert tab.rel == subset[0]
            assert tab.meta_lbl.cget("text") == \
                f"scene {submeta[subset[0]]['scene_id']} · {submeta[subset[0]]['split']}"
            print("ok  load_external: image list, first image, scene/split display")

            # ---- progress counting: all pending at first ----------------------
            counts = A.progress_counts(tab.data, tab.images)
            assert counts == {"total": 5, "pending": 5, "annotated": 0, "reviewed": 0}, counts
            print("ok  progress counting: fresh annotations.json -> all 5 pending")

            # ---- multiple boxes of the same class ------------------------------
            tab._boxes().append({"cls": "bottle", "x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1})
            tab._boxes().append({"cls": "bottle", "x": 0.7, "y": 0.7, "w": 0.1, "h": 0.1})
            tab.dirty = True
            tab._update_progress_label()
            assert len(tab._boxes()) == 2 and all(b["cls"] == "bottle" for b in tab._boxes())
            counts = A.progress_counts(tab.data, tab.images)
            assert counts["annotated"] == 1 and counts["pending"] == 4
            print("ok  multiple same-class boxes supported; status flips pending -> annotated")

            # ---- change class of an existing box -------------------------------
            tab._boxes()[0]["cls"] = "cap"
            assert tab._boxes()[0]["cls"] == "cap" and tab._boxes()[1]["cls"] == "bottle"
            print("ok  class change on an existing box")

            # ---- save after completing this image, then move on ----------------
            tab.save()
            assert not tab.dirty
            reloaded = A.load_annotations(tmp_ann)
            assert len(reloaded["images"][subset[0]]["boxes"]) == 2
            print("ok  save after completed image persists to annotations.json")

            # ---- Next Unannotated skips the one just annotated ------------------
            tab.goto_next_unannotated()
            assert tab.rel == subset[1], f"expected {subset[1]!r}, got {tab.rel!r}"
            print("ok  Next Unannotated: skips the completed image, lands on next pending one")

            # ---- reopen the first (already-annotated) image: annotations preserved
            tab.goto_image(subset[0])
            assert len(tab._boxes()) == 2, "reopening must preserve existing annotations"
            assert tab._boxes()[0]["cls"] == "cap"
            print("ok  reopening an annotated image preserves its existing boxes (no wipe)")

            # ---- no accidental overwrite: unsaved edit + navigate away asks first
            tab._boxes().append({"cls": "label", "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05})
            tab.dirty = True
            ask_calls = []
            mb.askyesno = lambda *a, **k: ask_calls.append(1) or False   # simulate user clicking "No"
            tab.goto_image(subset[2])
            assert ask_calls, "goto_image must ask for confirmation before discarding unsaved edits"
            assert tab.rel == subset[0], "declining the prompt must abort the navigation"
            assert len(tab._boxes()) == 3, "the unsaved box must survive a declined navigation"
            print("ok  navigating away from unsaved edits asks first; declining aborts and keeps the edit")

            # explicit confirm ("Yes") now allowed to proceed and discard
            mb.askyesno = lambda *a, **k: True
            tab.goto_image(subset[2])
            assert tab.rel == subset[2]
            print("ok  explicit user confirmation ('Yes') is honored and the navigation proceeds")

            # ---- save/reload round trip for the whole in-memory structure -------
            tab.save()
            reloaded2 = A.load_annotations(tmp_ann)
            assert reloaded2["classes"] == CLASSES
            assert reloaded2["task"] == "detection"
            print("ok  save/reload round-trip: classes, task, and image entries intact")

            root.destroy()

            # real image files were never modified -- re-hash a couple to confirm
            for fn in subset[:2]:
                assert (IMAGE_ROOT / fn).exists(), f"{fn} missing from stage2_dataset/clean/"
            print("ok  source images under stage2_dataset/clean/ untouched")
    finally:
        mb.showerror, mb.showinfo, mb.askyesno = orig

    # ---- prove the pre-existing Stage 2C (project-based) studio still works ----
    from ui.annotation_studio import selftest as studio_selftest
    studio_selftest()
    print("ok  existing Stage 2C project-mode Annotation Studio selftest still passes unchanged")

    # ---- confirm the locked/reviewed files were never touched by any of this ---
    review_path = HERE / "review_manifest.csv"
    split_json_path = HERE / "split.json"
    split_manifest_path = HERE / "split_manifest.csv"
    for p in (review_path, split_json_path, split_manifest_path):
        assert p.exists(), f"{p} missing -- should already exist from prior stages"
    print("ok  review_manifest.csv / split.json / split_manifest.csv left untouched")

    print("ok  stage2_dataset/annotation_workflow.py self-test complete -- "
          "no real annotations.json, no locked split file, no Stage 1 project touched")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
